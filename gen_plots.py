"""
gen_plots.py — New comparison figures for CAS experiment results.

Usage
-----
    python gen_plots.py [--db-path PATH] [--out-dir DIR] [--format FMT]

Options
-------
--db-path PATH   LMDB database path       (default: ./run/stats.lmdb)
--out-dir  DIR   Output directory         (default: ./run/plots)
--format   FMT   pdf | png | svg          (default: pdf)

Figures generated
-----------------
  hk_vs_deffuant.{fmt}
      3×2 grid comparing HK and Deffuant.
      Rows 0-1: grouped bar plots (x = dynamics, bars = recsys) for
                log-convergence time, final variance, closed triads,
                peak-count=1 fraction.
      Row 2: community count (x) vs modularity (y) scatter for HK and
             Deffuant separately. Colour = recsys, marker = α/q cond,
             opacity = repost rate.  Each point = condition-mean over reps.

  voter_vs_galam.{fmt}
      2×2 grid comparing Voter and Galam.
      Row 0: bar plots for final magnetization and log-convergence time.
      Row 1: community vs modularity scatter for Voter and Galam
             (Random + Structure only; Opinion-M9 not available for
             discrete models).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from plot_utils import (
    setup_style,
    load_dataframe,
    mean_sem,
    save_fig,
    DYNAMICS_LABEL,
    RECSYS_ORDER,
    RECSYS_LABEL,
    RECSYS_COLOR,
    AQ_ORDER,
    AQ_LABEL,
    AQ_MARKER,
    REPOST_ORDER,
    REPOST_LABEL,
    REPOST_ALPHA,
    _default_db,
    _default_out,
)


# ── Shared panel helpers ──────────────────────────────────────────────────────


def _bar_panel(
    ax: Axes,
    df: pd.DataFrame,
    dynamics_list: List[str],
    metric: str,
    ylabel: str,
    recsys_list: List[str],
    title: str = "",
) -> None:
    """Grouped bar chart: x = dynamics, bar groups = recsys (mean ± SEM over all reps/conditions)."""
    x = np.arange(len(dynamics_list))
    n_rec = len(recsys_list)
    bw = 0.22
    offsets = np.linspace(-(n_rec - 1) / 2, (n_rec - 1) / 2, n_rec) * bw

    for k, recsys in enumerate(recsys_list):
        rsub = df[df["recsys"] == recsys]
        for xi, dyn in enumerate(dynamics_list):
            if metric == "peak_count_is_1":
                vals = rsub[rsub["dynamics"] == dyn]["opinion_peak_count"]
                vals = (vals == 1).astype(float).dropna()
            else:
                vals = rsub[rsub["dynamics"] == dyn][metric].dropna()
            mu, se = mean_sem(vals)
            if np.isnan(mu):
                continue
            ax.bar(
                x[xi] + offsets[k],
                mu,
                bw * 0.9,
                color=RECSYS_COLOR[recsys],
                alpha=0.88,
                yerr=se,
                capsize=2.5,
                error_kw={"linewidth": 0.8, "capthick": 0.8},
            )

    ax.set_xticks(x)
    ax.set_xticklabels([DYNAMICS_LABEL[d] for d in dynamics_list], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=9)


def _apply_jitter(
    xs: np.ndarray,
    ys: np.ndarray,
    max_jitter: float = 0.3,
    n_iter: int = 300,
) -> np.ndarray:
    """Force-directed x-only jitter: push overlapping points apart with minimal displacement.

    Uses a soft repulsion potential between nearby points (defined by interaction
    radii r_x and r_y) plus a spring restoring force toward each point's original
    x position.  Only the x coordinate is modified; y (modularity) is unchanged.
    """
    xs_orig = np.asarray(xs, dtype=float).copy()
    n = len(xs_orig)
    if n <= 1:
        return xs_orig

    ys_arr = np.asarray(ys, dtype=float)
    y_range = max(float(np.ptp(ys_arr)), 1e-6)

    r_x = max_jitter * 0.4  # repulsion radius in x (data units)
    r_y = y_range * 0.05  # repulsion radius in y (5 % of y span)

    # Tiny seed jitter to break exact ties so coincident points can repel
    rng = np.random.default_rng(seed=0)
    xs_cur = xs_orig + rng.uniform(-r_x * 0.02, r_x * 0.02, n)
    xs_cur = np.clip(xs_cur, xs_orig - max_jitter, xs_orig + max_jitter)

    lr = 0.05
    spring_k = 0.05  # restoring spring — penalises displacement from original x

    for _ in range(n_iter):
        dx = xs_cur[:, None] - xs_cur[None, :]  # (n, n)
        dy = ys_arr[:, None] - ys_arr[None, :]  # (n, n)
        dist = np.sqrt((dx / r_x) ** 2 + (dy / r_y) ** 2)
        np.fill_diagonal(dist, np.inf)

        # Gradient of repulsion energy E = Σ max(0, 1-d)² / 2
        # dE/d(xs_i) = -Σ_j (1-d_ij) * dx_ij / (r_x² * d_ij)   [for d_ij < 1]
        overlap = np.maximum(0.0, 1.0 - dist)
        with np.errstate(invalid="ignore", divide="ignore"):
            rep_grad = np.where(
                dist < np.inf,
                -overlap * dx / (r_x**2 * (dist + 1e-9)),
                0.0,
            )

        # Total gradient = repulsion + spring toward original x
        gradients = rep_grad.sum(axis=1) + spring_k * (xs_cur - xs_orig)

        step = lr * gradients
        if np.max(np.abs(step)) < 1e-5:
            break

        xs_cur -= step
        xs_cur = np.clip(xs_cur, xs_orig - max_jitter, xs_orig + max_jitter)

    return xs_cur


def _community_modularity_scatter(
    ax: Axes,
    df: pd.DataFrame,
    dynamics: str,
    recsys_list: List[str],
    title: str,
) -> None:
    """Scatter: community count (x) vs modularity (y) for one dynamics model.

    Each point = per-(recsys, aq, repost) condition mean over reps.
    Colour = recsys, marker = α/q condition, opacity = repost rate.
    """
    sub = df[df["dynamics"] == dynamics]

    # ── First pass: collect every individual sample across all conditions ──────
    all_data: List[Tuple[np.ndarray, np.ndarray, str, str, str]] = []
    for recsys in recsys_list:
        rsub = sub[sub["recsys"] == recsys]
        for aq in AQ_ORDER:
            for repost in REPOST_ORDER:
                cell = rsub[(rsub["aq"] == aq) & (rsub["repost"] == repost)]
                xs = cell["community_count"].dropna()
                ys = cell["modularity"].dropna()
                valid = xs.index.intersection(ys.index)
                if len(valid) == 0:
                    continue
                all_data.append(
                    (xs.loc[valid].values, ys.loc[valid].values, recsys, aq, repost)  # type: ignore
                )

    if not all_data:
        return

    # ── Apply density-aware jitter to all points jointly ─────────────────────
    all_xs = np.concatenate([d[0] for d in all_data])
    all_ys = np.concatenate([d[1] for d in all_data])
    all_xs_j = _apply_jitter(all_xs, all_ys)

    # ── Second pass: scatter with jittered x positions ────────────────────────
    ptr = 0
    for xs_vals, ys_vals, recsys, aq, repost in all_data:
        n = len(xs_vals)
        xs_j = all_xs_j[ptr : ptr + n]
        ptr += n

        # Individual samples — small, semi-transparent, jittered x
        ax.scatter(
            xs_j,
            ys_vals,
            color=RECSYS_COLOR[recsys],
            marker=AQ_MARKER[aq],
            alpha=REPOST_ALPHA[repost] * 0.35,
            s=12,
            linewidths=0,
            zorder=2,
        )

        # Condition mean — large, opaque, original x (no jitter on the mean)
        ax.scatter(
            xs_vals.mean(),
            ys_vals.mean(),
            color=RECSYS_COLOR[recsys],
            marker=AQ_MARKER[aq],
            alpha=REPOST_ALPHA[repost],
            s=60,
            linewidths=0.4,
            edgecolors="white",
            zorder=3,
        )

    ax.set_xlabel("Community count", fontsize=9)
    ax.set_ylabel("Modularity", fontsize=9)
    ax.set_title(title, fontsize=9)


def _build_legend_handles(recsys_list: List[str]) -> list:
    """Legend handles: recsys colour patches, α/q marker shapes, repost opacity."""
    handles: list = []
    for r in recsys_list:
        handles.append(mpatches.Patch(color=RECSYS_COLOR[r], label=RECSYS_LABEL[r]))
    for aq in AQ_ORDER:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="0.35",
                marker=AQ_MARKER[aq],
                linestyle="None",
                markersize=6,
                label=AQ_LABEL[aq],
            )
        )
    for rp in REPOST_ORDER:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="0.35",
                marker="o",
                linestyle="None",
                markersize=6,
                alpha=REPOST_ALPHA[rp],
                label=REPOST_LABEL[rp],
            )
        )
    return handles


# ── Figure 1: HK vs Deffuant ─────────────────────────────────────────────────

_HK_DEFFUANT_BAR_METRICS: List[Tuple[str, str]] = [
    ("log_convergence_time", r"Log-conv. time $\log(1+t^*)$"),
    ("final_variance", "Final opinion variance"),
    ("n_closed_triangles", "Closed triads"),
    ("peak_count_is_1", r"Peak count $= 1$ fraction"),
]

_HK_DEFFUANT_RECSYS = ["random", "structure_m9", "opinion_m9"]
_HK_DEFFUANT_DYN = ["hk", "deffuant"]


def plot_hk_vs_deffuant(df: pd.DataFrame, out_dir: Path, fmt: str) -> None:
    """3×2 grid: 4 bar plots (rows 0–1) + 2 community/modularity scatters (row 2)."""
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    fig.suptitle("HK vs. Deffuant — metric comparison", fontsize=12, y=1.01)

    # ── Rows 0–1: bar plots ───────────────────────────────────────────────────
    for idx, (metric, ylabel) in enumerate(_HK_DEFFUANT_BAR_METRICS):
        ax = axes[idx // 2][idx % 2]
        _bar_panel(
            ax, df, _HK_DEFFUANT_DYN, metric, ylabel, _HK_DEFFUANT_RECSYS, title=ylabel
        )

    # ── Row 2: community vs modularity scatter ────────────────────────────────
    for col, dyn in enumerate(_HK_DEFFUANT_DYN):
        _community_modularity_scatter(
            axes[2][col],
            df,
            dyn,
            _HK_DEFFUANT_RECSYS,
            title=f"{DYNAMICS_LABEL[dyn]} — Community count vs Modularity",
        )

    handles = _build_legend_handles(_HK_DEFFUANT_RECSYS)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.03),
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_fig(fig, out_dir, "hk_vs_deffuant", fmt)


# ── Figure 2: Voter vs Galam ─────────────────────────────────────────────────

_VOTER_GALAM_BAR_METRICS: List[Tuple[str, str]] = [
    ("final_magnetization", r"Final magnetization $|\bar{x}|$"),
    ("log_convergence_time", r"Log-conv. time $\log(1+t^*)$"),
]

_VOTER_GALAM_RECSYS = ["random", "structure_m9"]
_VOTER_GALAM_DYN = ["voter", "galam"]


def plot_voter_vs_galam(df: pd.DataFrame, out_dir: Path, fmt: str) -> None:
    """2×2 grid: 2 bar plots (row 0) + 2 community/modularity scatters (row 1)."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    fig.suptitle("Voter vs. Galam — metric comparison", fontsize=12, y=1.01)

    # ── Row 0: bar plots ──────────────────────────────────────────────────────
    for col, (metric, ylabel) in enumerate(_VOTER_GALAM_BAR_METRICS):
        _bar_panel(
            axes[0][col],
            df,
            _VOTER_GALAM_DYN,
            metric,
            ylabel,
            _VOTER_GALAM_RECSYS,
            title=ylabel,
        )

    # ── Row 1: community vs modularity scatter ────────────────────────────────
    for col, dyn in enumerate(_VOTER_GALAM_DYN):
        _community_modularity_scatter(
            axes[1][col],
            df,
            dyn,
            _VOTER_GALAM_RECSYS,
            title=f"{DYNAMICS_LABEL[dyn]} — Community count vs Modularity",
        )

    handles = _build_legend_handles(_VOTER_GALAM_RECSYS)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.03),
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_fig(fig, out_dir, "voter_vs_galam", fmt)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparison scatter figures for CAS experiment results."
    )
    parser.add_argument(
        "--db-path",
        default=str(_default_db),
        metavar="PATH",
        help=f"LMDB database path (default: {_default_db})",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_default_out),
        metavar="DIR",
        help=f"Output directory (default: {_default_out})",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output figure format (default: pdf)",
    )
    args = parser.parse_args()

    setup_style()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from : {args.db_path}")
    df = load_dataframe(args.db_path)

    if df.empty:
        print(
            "No data found. Run run_stats.py first to populate the database.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded {len(df):,} records.  Generating figures → {args.out_dir}/\n")

    plot_hk_vs_deffuant(df, out_dir, args.format)
    plot_voter_vs_galam(df, out_dir, args.format)

    print(f"\nDone.  2 figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
