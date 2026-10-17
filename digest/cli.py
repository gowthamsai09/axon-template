"""CLI entry point. Phase 0: only `validate-profile` works; the rest are Phase 2 stubs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .profile import ProfileError, load_profile

STUBS = ("run", "init", "notify-failure")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="digest.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate-profile", help="validate profile.yaml")
    v.add_argument("path", nargs="?", type=Path, default=Path("profile.yaml"))
    for name in STUBS:
        sub.add_parser(name, help="not implemented until Phase 2").add_argument(
            "--state-dir", "--run-url", required=False
        )
    args, _ = ap.parse_known_args(argv)
    if args.cmd == "validate-profile":
        try:
            load_profile(args.path)
        except ProfileError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"{args.path}: OK")
        return 0
    print(f"digest.cli {args.cmd}: not implemented (Phase 2)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
