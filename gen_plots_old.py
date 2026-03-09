"""
gen_plots_old.py — Generate publication-quality figures for CAS experiment statistics.

Usage
-----
    python gen_plots_old.py [--db-path PATH] [--out-dir DIR] [--format FMT]

Options
-------
--db-path PATH   LMDB database path       (default: ./run/stats.lmdb)
--out-dir  DIR   Output directory         (default: ./run/plots)
--format   FMT   pdf | png | svg          (default: pdf)

Figures generated
-----------------
  log_convergence_time.{fmt}   — 2×2 grid  (AQ cond × repost) grouped-bar per metric
  final_variance.{fmt}
  final_magnetization.{fmt}
  community_count.{fmt}
  recsys_delta.{fmt}           — Relative change vs Random baseline (all metrics)
  condition_comparison.{fmt}   — Effect of α/q cond and repost rate
  summary_heatmap.{fmt}        — z-score heatmap (dynamics×recsys vs conditions)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless / script mode – must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from plot_utils import (
    setup_style,
    load_dataframe,
    mean_sem,
    save_fig,
    DYNAMICS_ORDER,
    DYNAMICS_LABEL,
    RECSYS_ORDER,
    RECSYS_LABEL,
    AQ_ORDER,
    AQ_LABEL,
    REPOST_ORDER,
    REPOST_LABEL,
    METRICS,
    RECSYS_COLOR,
    AQ_COLOR,
    REPOST_COLOR,
    _default_db,
    _default_out,
)


# ── Local aliases to preserve internal call sites ─────────────────────────────
def _setup_style() -> None:
    setup_style()


def _mean_sem(series: pd.Series) -> Tuple[float, float]:
    return mean_sem(series)


def _save(fig: Figure, out_dir: Path, name: str, fmt: str) -> None:
    save_fig(fig, out_dir, name, fmt)


# ── Figure 1-4: Per-metric 2×2 overview ───────────────────────────────────────
def plot_metric_overview(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_dir: Path,
    fmt: str,
) -> None:
    """
    2×2 subplot grid (rows = α/q condition, cols = repost rate).
    Each panel: grouped bar chart — X axis = dynamics model, bar groups = recsys type.
    Error bars show ±1 SEM across repetitions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharey=True, sharex=True)
    fig.suptitle(ylabel, fontsize=18, y=1.01)

    x = np.arange(len(DYNAMICS_ORDER))
    n_rec = len(RECSYS_ORDER)
    bw = 0.22
    offsets = np.linspace(-(n_rec - 1) / 2, (n_rec - 1) / 2, n_rec) * bw

    legend_handles = [
        mpatches.Patch(color=RECSYS_COLOR[r], label=RECSYS_LABEL[r])
        for r in RECSYS_ORDER
    ]

    for i, aq in enumerate(AQ_ORDER):
        for j, repost in enumerate(REPOST_ORDER):
            ax = axes[i][j]
            sub = df[(df["aq"] == aq) & (df["repost"] == repost)]

            for k, recsys in enumerate(RECSYS_ORDER):
                for xi, dyn in enumerate(DYNAMICS_ORDER):
                    vals = sub[(sub["recsys"] == recsys) & (sub["dynamics"] == dyn)][
                        metric
                    ]
                    mu, se = _mean_sem(vals)
                    if not np.isnan(mu):
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

            ax.set_title(f"{AQ_LABEL[aq]},  {REPOST_LABEL[repost]}", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([DYNAMICS_LABEL[d] for d in DYNAMICS_ORDER], fontsize=14)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=14)

    fig.legend(
        handles=legend_handles,
        title="Rec. system",
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, out_dir, metric, fmt)


# ── Figure 5: Recsys relative effect vs Random baseline ───────────────────────
def plot_recsys_delta(
    df: pd.DataFrame,
    out_dir: Path,
    fmt: str,
) -> None:
    """
    2×2 subplots (one per metric). For each metric, show the relative change
    (%) when using Structure-M9 or Opinion-M9 compared with Random, averaged
    over all α/q conditions and repost rates.
    Bar groups = dynamics model.
    """
    non_random = ["structure_m9", "opinion_m9"]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    fig.suptitle(
        "Effect of Recommendation System Relative to Random Baseline", fontsize=18
    )

    x = np.arange(len(DYNAMICS_ORDER))
    bw = 0.30
    offsets = np.array([-bw / 2, bw / 2])

    legend_handles = [
        mpatches.Patch(color=RECSYS_COLOR[r], label=RECSYS_LABEL[r]) for r in non_random
    ]

    for idx, (metric, ylabel) in enumerate(METRICS):
        ax = axes[idx // 2][idx % 2]

        baseline = df[df["recsys"] == "random"].groupby("dynamics")[metric].mean()

        for k, recsys in enumerate(non_random):
            alt = df[df["recsys"] == recsys].groupby("dynamics")[metric].mean()
            for xi, dyn in enumerate(DYNAMICS_ORDER):
                base = baseline.get(dyn, np.nan)
                val = alt.get(dyn, np.nan)
                if np.isnan(base) or np.isnan(val) or base == 0:
                    continue
                delta = (val - base) / abs(base) * 100.0
                ax.bar(
                    x[xi] + offsets[k],
                    delta,
                    bw * 0.9,
                    color=RECSYS_COLOR[recsys],
                    alpha=0.88,
                )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax.set_title(ylabel, fontsize=14)
        ax.set_ylabel("Relative change (%)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([DYNAMICS_LABEL[d] for d in DYNAMICS_ORDER], fontsize=14)

    fig.legend(
        handles=legend_handles,
        title="Rec. system",
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, out_dir, "recsys_delta", fmt)


# ── Figure 6: Condition comparison ────────────────────────────────────────────
def plot_condition_comparison(
    df: pd.DataFrame,
    out_dir: Path,
    fmt: str,
) -> None:
    """
    2-row × 4-column figure.
    Row 0: α/q condition effect on each metric (averaged over repost rates).
    Row 1: Repost rate effect on each metric (averaged over α/q conditions).
    X axis = dynamics model; bar groups = condition level; error bars = SEM.
    """
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharey=False)
    fig.suptitle("Effect of Interaction Conditions on Simulation Metrics", fontsize=18)

    x = np.arange(len(DYNAMICS_ORDER))
    bw = 0.32
    offsets = np.array([-bw / 2, bw / 2])

    # ── Row 0: α/q condition ─────────────────────────────────────────────────
    aq_handles = [
        mpatches.Patch(color=AQ_COLOR[a], label=AQ_LABEL[a]) for a in AQ_ORDER
    ]
    for col, (metric, ylabel) in enumerate(METRICS):
        ax = axes[0][col]
        for k, aq in enumerate(AQ_ORDER):
            sub = df[df["aq"] == aq].groupby("dynamics")[metric]
            for xi, dyn in enumerate(DYNAMICS_ORDER):
                mu, se = _mean_sem(
                    sub.get_group(dyn) if dyn in sub.groups else pd.Series(dtype=float)
                )
                if not np.isnan(mu):
                    ax.bar(
                        x[xi] + offsets[k],
                        mu,
                        bw * 0.9,
                        color=AQ_COLOR[aq],
                        alpha=0.88,
                        yerr=se,
                        capsize=2.5,
                        error_kw={"linewidth": 0.8, "capthick": 0.8},
                    )
        ax.set_title(ylabel, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([DYNAMICS_LABEL[d] for d in DYNAMICS_ORDER], fontsize=12)
        if col == 0:
            ax.set_ylabel("Mean value")

    # ── Row 1: Repost rate ────────────────────────────────────────────────────
    repost_handles = [
        mpatches.Patch(color=REPOST_COLOR[r], label=REPOST_LABEL[r])
        for r in REPOST_ORDER
    ]
    for col, (metric, ylabel) in enumerate(METRICS):
        ax = axes[1][col]
        for k, repost in enumerate(REPOST_ORDER):
            sub = df[df["repost"] == repost].groupby("dynamics")[metric]
            for xi, dyn in enumerate(DYNAMICS_ORDER):
                mu, se = _mean_sem(
                    sub.get_group(dyn) if dyn in sub.groups else pd.Series(dtype=float)
                )
                if not np.isnan(mu):
                    ax.bar(
                        x[xi] + offsets[k],
                        mu,
                        bw * 0.9,
                        color=REPOST_COLOR[repost],
                        alpha=0.88,
                        yerr=se,
                        capsize=2.5,
                        error_kw={"linewidth": 0.8, "capthick": 0.8},
                    )
        ax.set_title(ylabel, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([DYNAMICS_LABEL[d] for d in DYNAMICS_ORDER], fontsize=12)
        if col == 0:
            ax.set_ylabel("Mean value")

    # Row labels
    for row_idx, label in enumerate([r"$\alpha$/$q$ condition", "Repost rate"]):
        axes[row_idx][0].annotate(
            label,
            xy=(-0.38, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            rotation=90,
            fontsize=15,
            fontweight="bold",
        )

    # Separate legends per row, placed to the right
    axes[0][3].legend(
        handles=aq_handles,
        title=r"$\alpha$/$q$ cond.",
        loc="upper right",
        fontsize=12,
    )
    axes[1][3].legend(
        handles=repost_handles,
        title="Repost rate",
        loc="upper right",
        fontsize=12,
    )

    fig.tight_layout()
    _save(fig, out_dir, "condition_comparison", fmt)


# ── Figure 7: Summary heatmap ─────────────────────────────────────────────────
def plot_summary_heatmap(
    df: pd.DataFrame,
    out_dir: Path,
    fmt: str,
) -> None:
    """
    2×2 heatmap grid (one panel per metric).
    Rows = (dynamics, recsys) combinations (excluding invalid pairs).
    Cols = (α/q, repost) condition combinations.
    Cell colour encodes column-wise z-score; raw mean annotated in each cell.
    """
    # Valid (dynamics, recsys) pairs
    row_keys = [
        (d, r)
        for d in DYNAMICS_ORDER
        for r in RECSYS_ORDER
        if not (d in ("galam", "voter") and r == "opinion_m9")
    ]
    row_labels = [f"{DYNAMICS_LABEL[d]} / {RECSYS_LABEL[r]}" for d, r in row_keys]

    col_keys = [(aq, rp) for aq in AQ_ORDER for rp in REPOST_ORDER]
    col_labels = [f"{AQ_LABEL[aq]}\n{REPOST_LABEL[rp]}" for aq, rp in col_keys]

    n_rows, n_cols = len(row_keys), len(col_keys)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "Mean Metric Values by Dynamics × Recsys and Condition (column z-score)",
        fontsize=18,
    )

    for idx, (metric, ylabel) in enumerate(METRICS):
        ax = axes[idx // 2][idx % 2]

        mat = np.full((n_rows, n_cols), np.nan)
        for ri, (dyn, rec) in enumerate(row_keys):
            for ci, (aq, rp) in enumerate(col_keys):
                vals = df[
                    (df["dynamics"] == dyn)
                    & (df["recsys"] == rec)
                    & (df["aq"] == aq)
                    & (df["repost"] == rp)
                ][metric]
                if len(vals) > 0:
                    mat[ri, ci] = vals.mean()

        # Column-wise z-score for colour scale
        col_mean = np.nanmean(mat, axis=0)
        col_std = np.nanstd(mat, axis=0)
        col_std[col_std == 0] = 1.0
        mat_z = (mat - col_mean) / col_std

        im = ax.imshow(mat_z, aspect="auto", cmap="RdYlBu_r", vmin=-2.0, vmax=2.0)
        plt.colorbar(im, ax=ax, label="z-score", fraction=0.03, pad=0.02)

        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(col_labels, fontsize=11)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(row_labels, fontsize=11)
        ax.set_title(ylabel, fontsize=14)

        # Annotate each cell with the raw mean
        for ri in range(n_rows):
            for ci in range(n_cols):
                val = mat[ri, ci]
                if not np.isnan(val):
                    # choose text colour for contrast
                    z = mat_z[ri, ci]
                    txt_color = "white" if abs(z) > 1.4 else "black"
                    ax.text(
                        ci,
                        ri,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=txt_color,
                    )

    fig.tight_layout()
    _save(fig, out_dir, "summary_heatmap", fmt)


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots for CAS experiment results."
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

    _setup_style()

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

    # Figures 1–4: per-metric overview
    for metric, ylabel in METRICS:
        plot_metric_overview(df, metric, ylabel, out_dir, args.format)

    # Figure 5: recsys relative effect
    plot_recsys_delta(df, out_dir, args.format)

    # Figure 6: condition comparison
    plot_condition_comparison(df, out_dir, args.format)

    # Figure 7: summary heatmap
    plot_summary_heatmap(df, out_dir, args.format)

    print(f"\nDone.  7 figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
