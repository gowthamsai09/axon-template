"""Export JSON Schema from the pydantic models (NOT in the spec; see Spec deltas).

python -m scripts.export_schemas --out ../Axon/schemas     # write into the hub repo
python -m scripts.export_schemas --check ../Axon/schemas   # CI: fail if the hub copy differs
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from digest.profile import Profile
from digest.task_result import TaskResult

HUB_RAW = "https://raw.githubusercontent.com/gowthamsai09/Axon/main/schemas"
OWNED: dict[str, type[BaseModel]] = {
    "profile": Profile,
    "task-result": TaskResult,
}


def render(name: str, model: type[BaseModel]) -> str:
    schema: dict[str, Any] = model.model_json_schema(mode="validation")
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{HUB_RAW}/{name}.schema.json",
        **schema,
    }
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def export(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, model in OWNED.items():
        p = out / f"{name}.schema.json"
        p.write_text(render(name, model), encoding="utf-8")
        paths.append(p)
    return paths


def check(hub: Path) -> list[str]:
    problems: list[str] = []
    for name, model in OWNED.items():
        p = hub / f"{name}.schema.json"
        want = render(name, model)
        if not p.exists():
            problems.append(f"missing in hub: {p}")
        elif p.read_text(encoding="utf-8") != want:
            diff = difflib.unified_diff(
                p.read_text(encoding="utf-8").splitlines(),
                want.splitlines(),
                f"hub/{p.name}",
                f"exported/{p.name}",
                lineterm="",
            )
            problems.append("\n".join(diff))
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", type=Path)
    g.add_argument("--check", type=Path)
    args = ap.parse_args(argv)
    if args.out:
        for p in export(args.out):
            print(f"wrote {p}")
        return 0
    problems = check(args.check)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print("Schema drift: re-run with --out and commit to the hub repo.", file=sys.stderr)
        return 1
    print("schemas match the hub")
    return 0


if __name__ == "__main__":
    sys.exit(main())
