from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRECTORY_NAMES = {".git", ".venv", ".pytest_cache", "__pycache__", ".pio"}
FORBIDDEN = (
    "cardio" + "logist",
    "cardio" + "logo",
    "cardio" + "logía",
    "cardio" + "logia",
    "adjud" + "icat",
    "cur" + "at",
    "re" + "lab" + "el",
    "label_" + "update",
    "approved_" + "re" + "lab" + "el",
)


def test_public_language_excludes_private_review_terms():
    checked_suffixes = {".md", ".json", ".csv", ".py", ".ini", ".toml", ".cff"}
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in checked_suffixes:
            continue
        relative_parts = path.relative_to(ROOT).parts
        if any(part in IGNORED_DIRECTORY_NAMES for part in relative_parts[:-1]):
            continue
        if relative_parts[:2] == ("data", "cache"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for token in FORBIDDEN:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}:{token}")
    assert offenders == []
