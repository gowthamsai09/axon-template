"""Fail if this repo's prompts differ from the hub's canonical copies (spec 4.4, Appendix C 82).

python -m scripts.check_prompt_sync --hub ../Axon       exit 0 in sync, 1 drift, 2 hub copy missing
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from digest.summarizers.prompting import prompt_version

ROOT = Path(__file__).resolve().parent.parent
FILES = ("summarizer_system.txt", "onboarding_system.txt")


def _text(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def check(hub: Path, root: Path = ROOT) -> list[str]:
    """Problems found (empty = in sync). Messages carry file names and ids only."""
    prompts = hub / "prompts"
    problems: list[str] = []
    for name in FILES:
        theirs, mine = prompts / name, root / "digest" / "prompts" / name
        if not theirs.is_file():
            return [f"hub copy missing: {theirs}"]
        if _text(theirs) != _text(mine):
            digest = hashlib.sha256(_text(mine).encode()).hexdigest()[:10]
            problems.append(f"{name} differs from the hub copy (this repo: p-{digest})")
    version = prompts / "summarizer_system.version"
    if not version.is_file():
        return [f"hub copy missing: {version}"]
    mine_id = prompt_version(_text(root / "digest" / "prompts" / "summarizer_system.txt"))
    hub_id = version.read_text(encoding="utf-8").strip()
    if mine_id != hub_id:
        problems.append(f"prompt id {mine_id} differs from the hub's {hub_id}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hub", type=Path, required=True, help="path of the hub checkout")
    args = ap.parse_args(argv)
    problems = check(args.hub)
    for p in problems:
        print(f"prompt drift: {p}", file=sys.stderr)
    if any(p.startswith("hub copy missing") for p in problems):
        return 2
    print("prompts match the hub" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
