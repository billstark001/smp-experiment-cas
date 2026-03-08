"""
gen_opinion_trace_plots.py — Opinion trajectory plots for selected scenarios.

For each of the three criteria below, selects the first simulation run (by
rep index, then repost condition) where opinion_peak_count != 1, loads the
full opinion time-series, and plots every agent's opinion trajectory from
step 0 to 80 % of the total simulation length.

Criteria
--------
1. HK,       recsys = structure_m9, q > α  (aq = q_gt_a)
2. HK,       recsys = opinion_m9,   q > α
3. Deffuant, recsys = opinion_m9,   q > α

Usage
-----
    python gen_opinion_trace_plots.py [--db-path PATH] [--out-dir DIR] [--format FMT]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

# ── Locate smp_bindings ───────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent / "social-media-models"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from smp_bindings import RawSimulationRecord  # type: ignore[import]

from plot_utils import (
    setup_style,
    load_dataframe,
    save_fig,
    _default_db,
    _default_out,
    RECSYS_LABEL,
    AQ_LABEL,
    DYNAMICS_LABEL,
)

# ── Default base path for raw simulation data ─────────────────────────────────
_default_base = (
    Path(os.path.expanduser(os.environ["SMP_BASE_PATH"])).resolve()
    if "SMP_BASE_PATH" in os.environ
    else Path(__file__).resolve().parent / "run"
)

# ── Three selection criteria ──────────────────────────────────────────────────
CRITERIA = [
    {"dynamics": "hk", "recsys": "structure_m9", "aq": "q_gt_a"},
    {"dynamics": "hk", "recsys": "opinion_m9", "aq": "q_gt_a"},
    {"dynamics": "deffuant", "recsys": "opinion_m9", "aq": "q_gt_a"},
]

# Panel sub-labels
_PANEL_LABELS = ["(a)", "(b)", "(c)"]

# Trace style: single colour, very thin lines, low opacity for density
_TRACE_COLOR = "#2166ac"  # muted blue
_TRACE_LW = 0.25
_TRACE_ALPHA = 0.35

_STEP_CAP = 0.3

def _find_scenario(
    df: pd.DataFrame, dynamics: str, recsys: str, aq: str
) -> pd.Series | None:
    """Return the first row (sorted by rep, then repost) with peak_count != 1."""
    sub = df[
        (df["dynamics"] == dynamics)
        & (df["recsys"] == recsys)
        & (df["aq"] == aq)
        & (df["opinion_peak_count"] != 1)
    ].copy()

    if sub.empty:
        return None

    # Sort by rep first, then repost for determinism
    repost_order = {"p0": 0, "p25": 1}
    sub["_repost_ord"] = sub["repost"].map(repost_order).fillna(99)
    sub = sub.sort_values(["rep", "_repost_ord"])
    return sub.iloc[0]


def _unique_name(row: pd.Series) -> str:
    return f"{row['dynamics']}-{row['recsys']}-{row['aq']}-{row['repost']}-r{int(row['rep']):02d}"


def _plot_traces(ax: Axes, opinions: np.ndarray, cutoff_step: int) -> None:
    """Plot per-agent opinion traces up to *cutoff_step* on *ax*."""
    # opinions shape: (total_steps + 1, n_agents)
    steps = np.arange(cutoff_step + 1)
    traces = opinions[: cutoff_step + 1]  # (cutoff+1, agents)

    # Draw all agents in a single call per agent to avoid repeated overhead,
    # but batch via a 2-D transpose: each column is one agent's trajectory.
    ax.plot(
        steps,
        traces,  # (time, agents) → matplotlib broadcasts columns
        color=_TRACE_COLOR,
        linewidth=_TRACE_LW,
        alpha=_TRACE_ALPHA,
        rasterized=True,  # rasterise dense lines for smaller PDF/SVG output
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Draw opinion trajectory plots.")
    parser.add_argument("--db-path", default=str(_default_db), metavar="PATH")
    parser.add_argument("--base-path", default=str(_default_base), metavar="PATH")
    parser.add_argument("--out-dir", default=str(_default_out), metavar="DIR")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    args = parser.parse_args(argv)

    setup_style()

    df = load_dataframe(args.db_path)
    if df.empty:
        print("[ERROR] No data loaded from database.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build the figure: 1 row × 3 columns ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for ax, crit, panel_label in zip(axes, CRITERIA, _PANEL_LABELS):
        dynamics = crit["dynamics"]
        recsys = crit["recsys"]
        aq = crit["aq"]

        row = _find_scenario(df, dynamics, recsys, aq)
        if row is None:
            ax.set_visible(False)
            print(
                f"[WARN] No matching scenario for {dynamics}/{recsys}/{aq}",
                file=sys.stderr,
            )
            continue

        name = _unique_name(row)
        print(f"  Loading {name}  (peak_count={int(row['opinion_peak_count'])})")

        record = RawSimulationRecord(args.base_path, {"UniqueName": name})
        if not record.is_finished or not record.is_sanitized:
            print(f"[WARN] Record {name} is not finished/sanitised.", file=sys.stderr)
            ax.set_visible(False)
            continue

        with record:
            opinions = record.opinions.astype(np.float32)  # (steps+1, agents)
            max_step = record.max_step

        cutoff = int(max_step * _STEP_CAP)

        _plot_traces(ax, opinions, cutoff)

        # Axis decoration
        dyn_label = DYNAMICS_LABEL.get(dynamics, dynamics.upper())
        recsys_label = RECSYS_LABEL.get(recsys, recsys)
        aq_label = AQ_LABEL.get(aq, aq)
        peak_count = int(row["opinion_peak_count"])

        ax.set_title(
            f"{dyn_label} · {recsys_label} · {aq_label}\n"
            f"peaks = {peak_count},  rep {int(row['rep'])} ({row['repost']})",
            fontsize=9,
        )
        ax.set_xlabel("Simulation step", fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, cutoff)

        # Panel label
        ax.text(
            -0.10,
            1.06,
            panel_label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )

    axes[0].set_ylabel("Opinion", fontsize=9)

    save_fig(fig, out_dir, "fig_opinion_traces", args.format)
    print("Done.")


if __name__ == "__main__":
    main()
