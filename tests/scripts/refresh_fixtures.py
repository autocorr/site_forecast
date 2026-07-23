"""Unified, manual entry point for regenerating the offline test fixtures.

Each source fetches its upstream live, writes the fixture(s) atomically, and —
where the fixture can be signatured — prints a diff of the regenerated fixture
vs. the previously-committed one so dtype / index-resolution drift and
accidental shape changes are visible at a glance.

Manual invocation only: this touches the network and the VLA monitor DB, so it
never runs in CI. Run from the repo root, e.g.::

    uv run python tests/scripts/refresh_fixtures.py --list
    uv run python tests/scripts/refresh_fixtures.py ndfd sensitivity
    uv run python tests/scripts/refresh_fixtures.py --all

See ``tests/scripts/refresh/`` for the per-source library.
"""

from __future__ import annotations

import argparse
import sys

from refresh import FixtureSource
from refresh.herbie import SOURCE as HERBIE
from refresh.monitor import SOURCE as MONITOR
from refresh.ndfd import SOURCE as NDFD
from refresh.open_meteo import SOURCE as OPEN_METEO
from refresh.sensitivity import SOURCE as SENSITIVITY
from refresh import render_diff


SOURCES: tuple[FixtureSource, ...] = (
    MONITOR,
    NDFD,
    HERBIE,
    SENSITIVITY,
    OPEN_METEO,
)
BY_NAME = {s.name: s for s in SOURCES}


def _run(source: FixtureSource) -> None:
    print(f"\n=== refreshing {source.name} ===", flush=True)
    before = source.load() if source.load else None
    source.refresh()
    if source.load is not None and before is not None:
        after = source.load()
        print(render_diff(source.name, before, after), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="refresh_fixtures.py",
        description="Regenerate offline test fixtures (manual; hits network/DB).",
    )
    parser.add_argument(
        "names",
        nargs="*",
        metavar="SOURCE",
        help=f"sources to refresh (choose from: {', '.join(BY_NAME)})",
    )
    parser.add_argument("--all", action="store_true", help="refresh every source")
    parser.add_argument(
        "--list", action="store_true", help="list the registered sources and exit"
    )
    args = parser.parse_args(argv)

    if args.list:
        for s in SOURCES:
            kind = "diff" if s.load else "rewrite"
            print(f"{s.name:12s} ({kind})", flush=True)
        return 0

    if args.all:
        selected = list(SOURCES)
    elif args.names:
        unknown = [n for n in args.names if n not in BY_NAME]
        if unknown:
            parser.error(f"unknown source(s): {', '.join(unknown)}")
        selected = [BY_NAME[n] for n in args.names]
    else:
        # No target: never silently hit the network + DB for all five.
        parser.print_help()
        print(
            "\nGive source name(s), or --all. Nothing refreshed.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    for source in selected:
        _run(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
