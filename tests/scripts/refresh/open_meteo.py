"""Open-Meteo fixture source: the four VCR cassettes.

Unlike the snapshot sources, these fixtures are FlatBuffers responses recorded
by ``pytest-recording``, and the parser extracts a *fixed* column list — so a
fixture column-diff is inherently weak and upstream column drift is the live
``-m schema`` lane's job. This source therefore has no ``load()`` diff; its
deliverable is the re-recording:

1. ``pytest ... --record-mode=rewrite`` re-records against the live endpoints.
2. ``pytest ...`` replays the fresh cassettes; pandera ``strict=True`` in the
   offline tests fails on any structural drift, which is the verification.

Scoped to ``test_open_meteo.py`` so the whole suite isn't re-recorded.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import FixtureSource


REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_FILE = "tests/query/test_open_meteo.py"


def _pytest(*args: str) -> int:
    cmd = [sys.executable, "-m", "pytest", TEST_FILE, *args]
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def refresh() -> None:
    # Re-record. `rewrite` replaces existing cassettes (incl. the shared
    # `open_meteo_vla.yaml`); if that ever mishandles the shared cassette,
    # switch to `--record-mode=all`.
    rc = _pytest("--record-mode=rewrite")
    if rc != 0:
        raise RuntimeError(f"cassette re-recording failed (pytest exit {rc}).")

    # Replay the fresh cassettes offline; pandera strict schemas are the check.
    rc = _pytest("-p", "no:cacheprovider")
    if rc != 0:
        raise RuntimeError(
            f"offline replay of re-recorded cassettes failed (pytest exit {rc}); "
            "upstream response structure may have drifted — check the schemas."
        )
    print("[open_meteo] cassettes re-recorded; offline replay passes.", flush=True)


SOURCE = FixtureSource("open_meteo", refresh=refresh, load=None)
