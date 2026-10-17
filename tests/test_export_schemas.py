from __future__ import annotations

from pathlib import Path

from scripts.export_schemas import OWNED, check, export, main


def test_export_deterministic_and_checkable(tmp_path: Path) -> None:
    export(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {f"{n}.schema.json" for n in OWNED}
    assert check(tmp_path) == [] and main(["--check", str(tmp_path)]) == 0


def test_drift_and_missing(tmp_path: Path) -> None:
    assert len(check(tmp_path)) == len(OWNED)
    export(tmp_path)
    (tmp_path / "profile.schema.json").write_text("{}\n")
    assert main(["--check", str(tmp_path)]) == 1
