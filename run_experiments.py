"""
run_experiments.py — Launch all SMP experiment simulations.

Usage
-----
    python run_experiments.py [--dry-run] [--concurrency N] [--max-step S]

Options
-------
--dry-run         Print the parameter table and total count; do not run.
--concurrency N   Maximum parallel simulations (default: 4).
--max-step S      Override MaxSimulationStep for every scenario.

Output
------
Results are written to ./run/<UniqueName>/ by the Go binary.
Each completed simulation directory contains:
  acc-state-<step>.lz4    — opinion time-series (LZ4-compressed binary)
  graph-<step>.msgpack    — graph snapshots at regular intervals
  events.db               — SQLite event log (post / rewiring events)
  finished-*              — marker file written on clean completion

Metrics
------------------------------------------------------
- Average log-convergence time
- Final opinion cluster count (Leiden community detection)
- Opinion variance / magnetization
- Number of closed triangles in the final graph
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent / "social-media-models"

from smp_bindings import run_simulations, is_simulation_finished
from scenarios import generate_scenarios, print_summary

# region Paths
_default_binary = _REPO_ROOT / "smp"
_default_base = Path(__file__).resolve().parent / "run"

BINARY_PATH = (
    str(Path(os.path.expanduser(os.environ["SMP_BINARY_PATH"])).resolve())
    if "SMP_BINARY_PATH" in os.environ
    else str(_default_binary)
)
BASE_PATH = (
    str(Path(os.path.expanduser(os.environ["SMP_BASE_PATH"])).resolve())
    if "SMP_BASE_PATH" in os.environ
    else str(_default_base)
)
# endregion


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SMP experiment simulations.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scenario summary only; do not launch any processes.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        metavar="N",
        help="Maximum number of parallel simulations (default: 4).",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        metavar="S",
        help="Override MaxSimulationStep for every scenario.",
    )
    args = parser.parse_args()

    scenarios = generate_scenarios()

    # Optional step override
    if args.max_step is not None:
        for s in scenarios:
            s["MaxSimulationStep"] = args.max_step

    # Summary
    print_summary()
    already_done = sum(1 for s in scenarios if is_simulation_finished(BASE_PATH, s))
    print(f"Already finished: {already_done} / {len(scenarios)}")

    if args.dry_run:
        print("\n[dry-run] Listing first 10 scenario names:")
        for s in scenarios[:10]:
            print(" ", s["UniqueName"])
        if len(scenarios) > 10:
            print(f"  … and {len(scenarios) - 10} more.")
        return

    # Verify binary exists
    if not os.path.isfile(BINARY_PATH):
        print(f"ERROR: Go binary not found at {BINARY_PATH}", file=sys.stderr)
        print("Run:  cd ../social-media-models && go build -o smp .", file=sys.stderr)
        sys.exit(1)

    os.makedirs(BASE_PATH, exist_ok=True)

    completed = run_simulations(
        binary_path=BINARY_PATH,
        base_path=BASE_PATH,
        scenarios=scenarios,
        max_concurrent=args.concurrency,
        show_progress=None,  # auto-detect TTY
        skip_finished=True,
        show_position=True,
    )

    print(f"\nDone. Ran {len(completed)} new simulation(s).")
    total_finished = sum(1 for s in scenarios if is_simulation_finished(BASE_PATH, s))
    print(f"Total finished  : {total_finished} / {len(scenarios)}")


if __name__ == "__main__":
    main()
