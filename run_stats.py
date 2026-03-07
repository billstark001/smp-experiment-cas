"""
run_stats.py — Compute and persist statistics for all finished simulations.

Usage
-----
    python run_stats.py [--db-path PATH] [--concurrency N] [--force]

Options
-------
--db-path PATH     Path to the output LMDB database directory
                   (default: ./run/stats.lmdb).
--concurrency N    Worker threads for parallel stat computation (default: 4).
--force            Recompute and overwrite already-stored entries.

Output
------
An LMDB database whose keys are scenario UniqueName strings (UTF-8 bytes)
and whose values are msgpack-encoded dicts with the following fields:

    convergence_step        int     – step at which opinions converged
    log_convergence_time    float   – log1p(convergence_step)
    final_variance          float   – variance of opinions at the final step
    final_magnetization     float   – |mean opinion| at the final step
    n_closed_triangles      int     – closed directed triangles in final graph
    community_count         int     – number of Leiden communities
    community_sizes         str     – JSON {community_id: size}
"""

from __future__ import annotations
from stat_utils import compute_all_stats
from scenarios import generate_scenarios

from smp_bindings import (  # type: ignore[import]
    RawSimulationRecord,
    is_simulation_finished,
)

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import lmdb
import msgpack
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

# ── Locate the smp_bindings package ──────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent / "social-media-models"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── Paths ─────────────────────────────────────────────────────────────────────
_default_base = Path(__file__).resolve().parent / "run"
_default_db = (
    Path(os.path.expanduser(os.environ["SMP_DB_PATH"])).resolve()
    if "SMP_DB_PATH" in os.environ
    else _default_base / "stats.lmdb"
)

BASE_PATH = (
    str(Path(os.path.expanduser(os.environ["SMP_BASE_PATH"])).resolve())
    if "SMP_BASE_PATH" in os.environ
    else str(_default_base)
)


# ── Worker ────────────────────────────────────────────────────────────────────


def _process_scenario(
    base_path: str,
    scenario: Dict[str, Any],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Load one simulation record, compute stats, and return (name, stats | None)."""
    name = scenario["UniqueName"]
    try:
        record = RawSimulationRecord(base_path, scenario)
        if not record.is_finished or not record.is_sanitized:
            return name, None
        with record:
            stats = compute_all_stats(record)
        return name, stats
    except Exception as exc:
        print(f"\n[WARN] {name}: {exc}", file=sys.stderr)
        return name, None


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and store simulation statistics into an LMDB database."
    )
    parser.add_argument(
        "--db-path",
        default=str(_default_db),
        metavar="PATH",
        help=f"Output LMDB directory (default: {_default_db}).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        metavar="N",
        help="Worker threads for parallel computation (default: 4).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite already-stored entries.",
    )
    args = parser.parse_args()

    scenarios = generate_scenarios()
    finished = [s for s in scenarios if is_simulation_finished(BASE_PATH, s)]
    print(f"Finished simulations : {len(finished)} / {len(scenarios)}")

    if not finished:
        print("Nothing to process.")
        return

    os.makedirs(args.db_path, exist_ok=True)

    # LMDB: 10 GiB ceiling; grows lazily on most platforms
    env = lmdb.open(args.db_path, map_size=10 * 1024**3, max_dbs=0)

    # Determine which scenarios still need processing
    if not args.force:
        with env.begin() as txn:
            to_process = [
                s for s in finished if txn.get(s["UniqueName"].encode()) is None
            ]
        skipped = len(finished) - len(to_process)
        if skipped:
            print(f"Already stored       : {skipped}  (use --force to recompute)")
    else:
        to_process = finished

    print(f"To process           : {len(to_process)}")
    if not to_process:
        env.close()
        return

    ok = err = 0

    def _handle_result(name: str, stats) -> int:
        """Write stats to LMDB. Returns 1 on success, 0 on failure."""
        if stats is None:
            return 0
        value = msgpack.packb(stats, use_bin_type=True)
        with env.begin(write=True) as txn:
            txn.put(name.encode(), value)
        return 1

    if args.concurrency <= 1:
        with tqdm(total=len(to_process), unit="sim") as bar:
            for s in to_process:
                name, stats = _process_scenario(BASE_PATH, s)
                bar.set_postfix_str(name[-45:])
                bar.update()
                if _handle_result(name, stats):
                    ok += 1
                else:
                    err += 1
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_process_scenario, BASE_PATH, s): s for s in to_process}
            with tqdm(total=len(futures), unit="sim") as bar:
                for future in as_completed(futures):
                    try:
                        # Timeout prevents the main thread from blocking indefinitely if a
                        # worker hangs (e.g. due to a pathological graph in compute_all_stats).
                        name, stats = future.result(timeout=300)
                    except FuturesTimeoutError:
                        scenario = futures[future]
                        name = scenario["UniqueName"]
                        print(f"\n[WARN] {name}: timed out after 300 s", file=sys.stderr)
                        bar.update()
                        err += 1
                        continue
                    bar.set_postfix_str(name[-45:])
                    bar.update()
                    if _handle_result(name, stats):
                        ok += 1
                    else:
                        err += 1

    env.close()
    print(f"\nStored : {ok}   Errors : {err}")


if __name__ == "__main__":
    main()
