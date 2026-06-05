"""Download ECG windows from TITAN V4 Edge and persist review artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
from scipy.io import savemat


DEFAULT_BASE_URL = "http://192.168.1.100"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "GRABACIONES"
DERIVED_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF")


def fetch_text(url: str, timeout: float = 5.0) -> str:
    with urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def fetch_json(url: str, timeout: float = 5.0) -> dict:
    return json.loads(fetch_text(url, timeout=timeout))


def parse_window_csv(raw_csv: str) -> dict:
    rows = list(csv.DictReader(io.StringIO(raw_csv)))
    if not rows:
        raise ValueError("The downloaded ECG window is empty.")

    cycle = int(rows[0]["cycle"])
    if any(int(row["cycle"]) != cycle for row in rows):
        raise ValueError("The downloaded ECG window mixes acquisition cycles.")

    samples = np.asarray(
        [
            [
                int(row["lead_i_adc"]),
                int(row["lead_ii_adc"]),
                int(row["lead_iii_adc"]),
            ]
            for row in rows
        ],
        dtype=np.int16,
    ).T
    return {"cycle": cycle, "raw_adc": samples}


def derive_six_leads(raw_adc: np.ndarray) -> np.ndarray:
    lead_i = raw_adc[0].astype(np.int32)
    lead_ii = raw_adc[1].astype(np.int32)
    lead_iii = lead_ii - lead_i
    avr = -(lead_i + lead_ii) / 2.0
    avl = lead_i - lead_ii / 2.0
    avf = lead_ii - lead_i / 2.0
    return np.rint(np.vstack([lead_i, lead_ii, lead_iii, avr, avl, avf])).astype(np.int16)


def write_window_artifacts(
    output_dir: Path,
    window: dict,
    quality: dict,
    sample_rate_hz: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cycle = int(window["cycle"])
    record_id = f"ecg_cycle_{cycle:06d}"
    raw_adc = np.asarray(window["raw_adc"], dtype=np.int16)
    derived = derive_six_leads(raw_adc)
    sample_count = raw_adc.shape[1]

    csv_path = output_dir / f"{record_id}.csv"
    mat_path = output_dir / f"{record_id}.mat"
    hea_path = output_dir / f"{record_id}.hea"
    metadata_path = output_dir / f"{record_id}.json"

    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_index",
                "lead_i_adc",
                "lead_ii_adc",
                "lead_iii_sensor_adc",
                "lead_i_derived",
                "lead_ii_derived",
                "lead_iii_derived",
                "avr_derived",
                "avl_derived",
                "avf_derived",
            ]
        )
        for sample in range(sample_count):
            writer.writerow([sample, *raw_adc[:, sample], *derived[:, sample]])

    savemat(
        mat_path,
        {
            "val": derived,
            "raw_adc": raw_adc,
            "fs": np.asarray([[sample_rate_hz]], dtype=np.int32),
            "cycle": np.asarray([[cycle]], dtype=np.int32),
            "lead_names": np.asarray(DERIVED_LEADS, dtype=object),
        },
    )

    header_lines = [f"{record_id} 6 {sample_rate_hz} {sample_count}"]
    for lead_name, first_value in zip(DERIVED_LEADS, derived[:, 0]):
        header_lines.append(
            f"{record_id}.mat 16 1/ADC 16 0 {int(first_value)} 0 0 {lead_name}"
        )
    header_lines.extend(
        [
            "# Signals are ADC-count representations and are not calibrated in mV.",
            "# Physical sensor channels are preserved in the paired MAT and CSV files.",
        ]
    )
    hea_path.write_text("\n".join(header_lines) + "\n", encoding="ascii")

    metadata = {
        "record_id": record_id,
        "cycle": cycle,
        "captured_at": datetime.now().astimezone().isoformat(),
        "sample_rate_hz": sample_rate_hz,
        "samples": sample_count,
        "duration_seconds": sample_count / sample_rate_hz,
        "physical_channels": ["I_ADC", "II_ADC", "III_SENSOR_ADC"],
        "derived_leads": list(DERIVED_LEADS),
        "quality": quality,
        "notes": [
            "Values are raw ADC counts; no mV calibration has been applied.",
            "Derived Lead III follows the firmware equation II - I.",
            "The third physical sensor channel is preserved separately for audit.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "csv": csv_path,
        "mat": mat_path,
        "hea": hea_path,
        "metadata": metadata_path,
    }


def create_session_dir(output_root: Path, label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_root / f"{timestamp}_{label}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def record_session(
    base_url: str,
    output_root: Path,
    duration_seconds: int,
    max_windows: int,
    label: str,
) -> Path:
    session_dir = create_session_dir(output_root, label)
    deadline = time.monotonic() + duration_seconds
    seen_cycles: set[int] = set()

    print(f"[RECORDER] Session: {session_dir}")
    print(f"[RECORDER] Source:  {base_url}")

    while time.monotonic() < deadline:
        try:
            quality = fetch_json(f"{base_url}/api/quality", timeout=4.0)
            cycle = int(quality.get("cycle", 0))
            if quality.get("window_ready") and cycle not in seen_cycles:
                raw_csv = fetch_text(f"{base_url}/api/window.csv", timeout=8.0)
                window = parse_window_csv(raw_csv)
                artifacts = write_window_artifacts(
                    output_dir=session_dir,
                    window=window,
                    quality=quality,
                    sample_rate_hz=int(quality["capture_rate_hz"]),
                )
                seen_cycles.add(cycle)
                status = "accepted" if quality.get("window_accepted") else "rejected"
                print(f"[RECORDER] cycle={cycle} quality={status} csv={artifacts['csv'].name}")
                if max_windows and len(seen_cycles) >= max_windows:
                    break
        except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            print(f"[RECORDER] Waiting for device: {exc}")
        time.sleep(1.0)

    summary = {
        "base_url": base_url,
        "started_session": session_dir.name,
        "saved_cycles": sorted(seen_cycles),
        "saved_window_count": len(seen_cycles),
    }
    (session_dir / "session_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"[RECORDER] Saved windows: {len(seen_cycles)}")
    return session_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--label", default="short_test")
    args = parser.parse_args()

    record_session(
        base_url=args.base_url.rstrip("/"),
        output_root=args.output_dir,
        duration_seconds=args.duration_seconds,
        max_windows=args.max_windows,
        label=args.label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
