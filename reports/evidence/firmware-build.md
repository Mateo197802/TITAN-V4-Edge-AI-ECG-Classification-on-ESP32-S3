# ESP32-S3 Firmware Build Evidence

Build date: 2026-09-24. This is a compile record only; no board was connected or flashed.

## Command and pinned toolchain

```powershell
python -m pip install -r requirements-firmware.txt
python -m platformio run -d hardware/esp32/firmware -e esp32s3
```

| Component | Version / identity |
|---|---|
| PlatformIO Core | 6.2.0 |
| Espressif32 platform | 6.12.0 |
| Arduino-ESP32 framework | 2.0.17 (`3.20017.241212+sha.dcc1105b`) |
| TensorFlowLite_ESP32 | `e88e0ebee0430ed716ff5b49854795db90066e59` |
| Target | ESP32-S3-DevKitC-1-N8R8 (project-local PlatformIO board definition) |
| Profile | `esp32s3` (sensitive HTTP routes disabled) |

## Result

PlatformIO reported `SUCCESS` for the full compile and binary-generation step.

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| RAM | 137,232 bytes | 327,680 bytes | 41.9% |
| Application partition | 5,517,309 bytes | 6,553,600 bytes | 84.2% |

Generated local binary SHA-256: `94390680B91A6C2A4CD965A4FA7D3B4A96E9DF5283C37DEEFB03DA3DA267AF56`. Build outputs are excluded from Git; the clean-checkout command above regenerates them.

The repository now pins the missing TFLite dependency and no longer sets `TF_LITE_STATIC_MEMORY`, which caused upstream placement-new code to fail due to a private class `operator delete`. The wrapper also no longer deletes the function-static interpreter as if it owned it. The firmware uses the preallocated tensor arena; these changes do not constitute a runtime or memory-safety validation on hardware.

The build emitted C compiler warnings from the pinned dependency's bundled LCD controller drivers, including implicit declarations for `LCD_WRITE_REG` and related macros. The firmware linked successfully; these third-party warnings were not suppressed or represented as resolved.

The dependency's upstream maintainer marks the library outdated and no longer recommended. It is pinned to preserve this revision's existing Arduino API and make its build reproducible. Migration to Espressif's maintained TFLite Micro component remains future work. No ECG files are downloaded or embedded by this firmware build, and no performance or clinical-safety claim follows from compilation.
