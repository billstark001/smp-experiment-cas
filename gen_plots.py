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

Outputs
-------
  fig_hk_deffuant_bars_v1.{fmt}
      1×4 row: 4 bar metrics (averaged over all aq/repost conditions).
  fig_hk_deffuant_bars_v2.{fmt}
      2×2 grid: same 4 bar metrics, each with 12 bars (x = dynamics×aq,
      bars = recsys), averaged over repost conditions.
  fig_hk_deffuant_scatter.{fmt}
      1×2 community-count vs modularity scatters for HK and Deffuant.
  tab_voter_galam_metrics.txt
      LaTeX table: mean ± std for final_magnetization and
      log_convergence_time for Voter and Galam (random + structure_m9).
  fig_voter_galam_scatter.{fmt}
      1×2 community-count vs modularity scatters for Voter and Galam.
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


# ── Subplot labeling ──────────────────────────────────────────────────────────


def _label_axes(axes_flat: List[Axes], start: int = 0) -> None:
    """Add bold (a), (b), … labels to each axis in left-to-right order."""
    for i, ax in enumerate(axes_flat):
        label = chr(ord("a") + start + i)
        ax.text(
            -0.12,
            1.06,
            f"({label})",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
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


def _bar_panel_aq_split(
    ax: Axes,
    df: pd.DataFrame,
    dynamics_list: List[str],
    aq_list: List[str],
    metric: str,
    ylabel: str,
    recsys_list: List[str],
    title: str = "",
) -> None:
    """Grouped bar chart with 12 bars: x = (dynamics, aq) combos, bar groups = recsys.

    x positions are ordered as: for each dynamics, all aq conditions;  with a
    small gap inserted between different dynamics groups.
    """
    # Build ordered x-group list: [(dyn, aq), ...]
    groups = [(dyn, aq) for dyn in dynamics_list for aq in aq_list]
    n_groups = len(groups)  # 2 * 2 = 4
    n_rec = len(recsys_list)  # 3
    bw = 0.18
    offsets = np.linspace(-(n_rec - 1) / 2, (n_rec - 1) / 2, n_rec) * bw

    # Insert a wider gap between dynamics groups
    gap = 0.35  # extra spacing between dynamics blocks
    x_positions: List[float] = []
    for gi, (dyn, aq) in enumerate(groups):
        dyn_idx = dynamics_list.index(dyn)
        within_idx = aq_list.index(aq)
        x_positions.append(dyn_idx * (len(aq_list) + gap) + within_idx)

    for k, recsys in enumerate(recsys_list):
        rsub = df[df["recsys"] == recsys]
        for gi, (dyn, aq) in enumerate(groups):
            cell = rsub[(rsub["dynamics"] == dyn) & (rsub["aq"] == aq)]
            if metric == "peak_count_is_1":
                vals = cell["opinion_peak_count"]
                vals = (vals == 1).astype(float).dropna()
            else:
                vals = cell[metric].dropna()
            mu, se = mean_sem(vals)
            if np.isnan(mu):
                continue
            ax.bar(
                x_positions[gi] + offsets[k],
                mu,
                bw * 0.9,
                color=RECSYS_COLOR[recsys],
                alpha=0.88,
                yerr=se,
                capsize=2.0,
                error_kw={"linewidth": 0.7, "capthick": 0.7},
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [f"{DYNAMICS_LABEL[d]}\n{AQ_LABEL[a]}" for d, a in groups],
        fontsize=8,
    )
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


# ── Figure 1a: HK vs Deffuant — bars v1 (1×4, aggregated over aq/repost) ────

_HK_DEFFUANT_BAR_METRICS: List[Tuple[str, str]] = [
    ("log_convergence_time", r"Log-conv. time $\log(1+t^*)$"),
    ("final_variance", "Final opinion variance"),
    ("n_closed_triangles", "Closed triads"),
    ("peak_count_is_1", r"Peak count $= 1$ fraction"),
]

_HK_DEFFUANT_RECSYS = ["random", "structure_m9", "opinion_m9"]
_HK_DEFFUANT_DYN = ["hk", "deffuant"]


def plot_hk_deffuant_bars_v1(df: pd.DataFrame, out_dir: Path, fmt: str) -> None:
    """1×4 row: 4 bar-metric panels averaged over all aq/repost conditions."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("HK vs. Deffuant — bar metrics", fontsize=12, y=1.02)

    for idx, (metric, ylabel) in enumerate(_HK_DEFFUANT_BAR_METRICS):
        _bar_panel(
            axes[idx],
            df,
            _HK_DEFFUANT_DYN,
            metric,
            ylabel,
            _HK_DEFFUANT_RECSYS,
            title=ylabel,
        )

    _label_axes(list(axes))

    handles = [mpatches.Patch(color=RECSYS_COLOR[r], label=RECSYS_LABEL[r]) for r in _HK_DEFFUANT_RECSYS]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.08),
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_fig(fig, out_dir, "fig_hk_deffuant_bars_v1", fmt)


# ── Figure 1b: HK vs Deffuant — bars v2 (2×2, 12 bars per subplot, split aq) ─


def plot_hk_deffuant_bars_v2(df: pd.DataFrame, out_dir: Path, fmt: str) -> None:
    """2×2 grid: 4 bar-metric panels, each with 12 bars (dynamics×aq on x, recsys bars)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("HK vs. Deffuant — bar metrics split by α/q condition", fontsize=12, y=1.02)

    for idx, (metric, ylabel) in enumerate(_HK_DEFFUANT_BAR_METRICS):
        ax = axes[idx // 2][idx % 2]
        _bar_panel_aq_split(
            ax,
            df,
            _HK_DEFFUANT_DYN,
            AQ_ORDER,
            metric,
            ylabel,
            _HK_DEFFUANT_RECSYS,
            title=ylabel,
        )

    _label_axes([axes[r][c] for r in range(2) for c in range(2)])

    handles = [mpatches.Patch(color=RECSYS_COLOR[r], label=RECSYS_LABEL[r]) for r in _HK_DEFFUANT_RECSYS]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.04),
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, out_dir, "fig_hk_deffuant_bars_v2", fmt)


# ── Figure 2: HK vs Deffuant — community/modularity scatter ─────────────────


def plot_hk_deffuant_scatter(df: pd.DataFrame, out_dir: Path, fmt: str) -> None:
    """1×2 community-count vs modularity scatter for HK and Deffuant."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("HK vs. Deffuant — Community count vs Modularity", fontsize=12, y=1.02)

    for col, dyn in enumerate(_HK_DEFFUANT_DYN):
        _community_modularity_scatter(
            axes[col],
            df,
            dyn,
            _HK_DEFFUANT_RECSYS,
            title=f"{DYNAMICS_LABEL[dyn]}",
        )

    _label_axes(list(axes))

    handles = _build_legend_handles(_HK_DEFFUANT_RECSYS)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.06),
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_fig(fig, out_dir, "fig_hk_deffuant_scatter", fmt)


# ── Table 1: Voter vs Galam — LaTeX metrics table ────────────────────────────

_VOTER_GALAM_TABLE_METRICS: List[Tuple[str, str]] = [
    ("final_magnetization", r"Final magnetization $|\bar{x}|$"),
    ("log_convergence_time", r"Log-conv. time $\log(1+t^*)$"),
]

_VOTER_GALAM_RECSYS = ["random", "structure_m9"]
_VOTER_GALAM_DYN = ["voter", "galam"]


def make_voter_galam_table(df: pd.DataFrame, out_dir: Path) -> None:
    """Write a LaTeX table (tab_voter_galam_metrics.txt) with mean±std.

    Rows: (dynamics, recsys); Columns: metrics.
    """
    lines: List[str] = []

    # ── header ────────────────────────────────────────────────────────────────
    n_metrics = len(_VOTER_GALAM_TABLE_METRICS)
    col_spec = "ll" + "r" * n_metrics
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Voter vs.~Galam: mean~$\pm$~std over all conditions}")
    lines.append(r"  \label{tab:voter_galam_metrics}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    header_cols = " & ".join(
        ["Dynamics", "Rec.~Sys."] + [label for _, label in _VOTER_GALAM_TABLE_METRICS]
    )
    lines.append(f"    {header_cols} \\\\")
    lines.append(r"    \midrule")

    # ── data rows ─────────────────────────────────────────────────────────────
    for dyn in _VOTER_GALAM_DYN:
        dsub = df[df["dynamics"] == dyn]
        for ri, recsys in enumerate(_VOTER_GALAM_RECSYS):
            rsub = dsub[dsub["recsys"] == recsys]
            dyn_cell = DYNAMICS_LABEL[dyn] if ri == 0 else ""
            rec_cell = RECSYS_LABEL[recsys]
            metric_cells: List[str] = []
            for metric, _ in _VOTER_GALAM_TABLE_METRICS:
                vals = rsub[metric].dropna()
                if len(vals) == 0:
                    metric_cells.append("---")
                else:
                    mu = vals.mean()
                    std = vals.std()
                    metric_cells.append(f"${mu:.3f} \\pm {std:.3f}$")
            row = " & ".join([dyn_cell, rec_cell] + metric_cells)
            lines.append(f"    {row} \\\\")
        lines.append(r"    \addlinespace")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    out_path = out_dir / "tab_voter_galam_metrics.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Wrote table → {out_path}")


# ── Figure 3: Voter vs Galam — community/modularity scatter ──────────────────


def plot_voter_galam_scatter(df: pd.DataFrame, out_dir: Path, fmt: str) -> None:
    """1×2 community-count vs modularity scatter for Voter and Galam."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Voter vs. Galam — Community count vs Modularity", fontsize=12, y=1.02)

    for col, dyn in enumerate(_VOTER_GALAM_DYN):
        _community_modularity_scatter(
            axes[col],
            df,
            dyn,
            _VOTER_GALAM_RECSYS,
            title=f"{DYNAMICS_LABEL[dyn]}",
        )

    _label_axes(list(axes))

    handles = _build_legend_handles(_VOTER_GALAM_RECSYS)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.06),
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_fig(fig, out_dir, "fig_voter_galam_scatter", fmt)


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

    print(f"Loaded {len(df):,} records.  Generating outputs → {args.out_dir}/\n")

    plot_hk_deffuant_bars_v1(df, out_dir, args.format)
    plot_hk_deffuant_bars_v2(df, out_dir, args.format)
    plot_hk_deffuant_scatter(df, out_dir, args.format)
    make_voter_galam_table(df, out_dir)
    plot_voter_galam_scatter(df, out_dir, args.format)

    print(f"\nDone.  5 outputs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
