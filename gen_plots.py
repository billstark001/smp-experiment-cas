"""
gen_plots.py — Generate publication-quality figures for CAS experiment statistics.

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
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import lmdb
import msgpack
import numpy as np
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")  # headless / script mode – must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

# ── Default paths ──────────────────────────────────────────────────────────────
_script_dir = Path(__file__).resolve().parent
_default_db = (
    Path(os.path.expanduser(os.environ["SMP_DB_PATH"])).resolve()
    if "SMP_DB_PATH" in os.environ
    else _script_dir / "run" / "stats.lmdb"
)
_default_out = (
    Path(os.path.expanduser(os.environ["SMP_PLOTS_PATH"])).resolve()
    if "SMP_PLOTS_PATH" in os.environ
    else _script_dir / "run" / "plots"
)

# ── Academic style setup ───────────────────────────────────────────────────────
_SERIF_PREF = [
    "Times New Roman",
    "Times",
    "Liberation Serif",
    "CMU Serif",
    "STIXGeneral",
    "DejaVu Serif",
]


def _setup_style() -> None:
    """Configure matplotlib for publication-quality academic figures."""
    # Prefer scienceplots if installed (pip install scienceplots)
    try:
        import scienceplots  # type: ignore[import]  # noqa: F401

        plt.style.use(["science", "no-latex"])
        return
    except ImportError:
        pass

    # Fallback: manual rcParams with best available serif font
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    serif = next((f for f in _SERIF_PREF if f in available), "DejaVu Serif")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [serif],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titlepad": 6,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.framealpha": 0.85,
            "legend.edgecolor": "0.7",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )


# ── Ordered sets & display labels ─────────────────────────────────────────────
DYNAMICS_ORDER = ["hk", "deffuant", "galam", "voter"]
DYNAMICS_LABEL = {
    "hk": "HK",
    "deffuant": "Deffuant",
    "galam": "Galam",
    "voter": "Voter",
}

RECSYS_ORDER = ["random", "structure_m9", "opinion_m9"]
RECSYS_LABEL = {
    "random": "Random",
    "structure_m9": "Structure (M9)",
    "opinion_m9": "Opinion (M9)",
}

AQ_ORDER = ["a_gt_q", "q_gt_a"]
AQ_LABEL = {"a_gt_q": r"$\alpha > q$", "q_gt_a": r"$q > \alpha$"}

REPOST_ORDER = ["p25", "p0"]
REPOST_LABEL = {"p25": r"$p = 0.25$", "p0": r"$p = 0.00$"}

METRICS: List[Tuple[str, str]] = [
    ("log_convergence_time", r"Log-convergence time $\log(1+t^*)$"),
    ("final_variance", "Final opinion variance"),
    ("final_magnetization", r"Final magnetization $|\bar{x}|$"),
    ("community_count", "Community count"),
]

# Okabe–Ito colorblind-friendly palette
_OI = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
]

RECSYS_COLOR = {
    "random": _OI[1],  # sky blue
    "structure_m9": _OI[5],  # vermillion
    "opinion_m9": _OI[2],  # bluish green
}

AQ_COLOR = {"a_gt_q": _OI[0], "q_gt_a": _OI[4]}
REPOST_COLOR = {"p25": _OI[5], "p0": _OI[7]}

# ── LMDB record key pattern ────────────────────────────────────────────────────
_NAME_RE = re.compile(
    r"^(?P<dynamics>hk|deffuant|galam|voter)"
    r"-(?P<recsys>random|structure_m9|opinion_m9)"
    r"-(?P<aq>a_gt_q|q_gt_a)"
    r"-(?P<repost>p\d+)"
    r"-r(?P<rep>\d+)$"
)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_dataframe(db_path: str) -> pd.DataFrame:
    """Load all LMDB stats records into a tidy pandas DataFrame."""
    if not os.path.isdir(db_path):
        print(f"[WARN] Database not found: {db_path}", file=sys.stderr)
        return pd.DataFrame()

    env = lmdb.open(db_path, readonly=True, lock=False)
    rows = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for raw_key, raw_val in cursor.iternext(keys=True, values=True):
            key = raw_key.decode()
            m = _NAME_RE.match(key)
            if m is None:
                continue
            data = msgpack.unpackb(raw_val, raw=False)
            row = dict(m.groupdict())
            for field, _ in METRICS:
                row[field] = data.get(field, np.nan)
            row["n_closed_triangles"] = data.get("n_closed_triangles", np.nan)
            rows.append(row)
    env.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["rep"] = df["rep"].astype(int)
    for field, _ in METRICS:
        df[field] = pd.to_numeric(df[field], errors="coerce")
    df["n_closed_triangles"] = pd.to_numeric(df["n_closed_triangles"], errors="coerce")
    return df


# ── Small helpers ──────────────────────────────────────────────────────────────
def _mean_sem(series: pd.Series) -> Tuple[float, float]:
    vals = series.dropna()
    if len(vals) == 0:
        return np.nan, 0.0
    return float(vals.mean()), float(vals.sem())  # type: ignore[arg-type]


def _save(fig: Figure, out_dir: Path, name: str, fmt: str) -> None:
    out = out_dir / f"{name}.{fmt}"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


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
    fig.suptitle(ylabel, fontsize=12, y=1.01)

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

            ax.set_title(f"{AQ_LABEL[aq]},  {REPOST_LABEL[repost]}", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels([DYNAMICS_LABEL[d] for d in DYNAMICS_ORDER], fontsize=9)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=9)

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
        "Effect of Recommendation System Relative to Random Baseline", fontsize=12
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
        ax.set_title(ylabel, fontsize=9)
        ax.set_ylabel("Relative change (%)", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([DYNAMICS_LABEL[d] for d in DYNAMICS_ORDER], fontsize=9)

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
    fig.suptitle("Effect of Interaction Conditions on Simulation Metrics", fontsize=12)

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
        ax.set_title(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([DYNAMICS_LABEL[d] for d in DYNAMICS_ORDER], fontsize=8)
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
        ax.set_title(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([DYNAMICS_LABEL[d] for d in DYNAMICS_ORDER], fontsize=8)
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
            fontsize=10,
            fontweight="bold",
        )

    # Separate legends per row, placed to the right
    axes[0][3].legend(
        handles=aq_handles,
        title=r"$\alpha$/$q$ cond.",
        loc="upper right",
        fontsize=8,
    )
    axes[1][3].legend(
        handles=repost_handles,
        title="Repost rate",
        loc="upper right",
        fontsize=8,
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
        fontsize=12,
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
        ax.set_xticklabels(col_labels, fontsize=7)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(ylabel, fontsize=9)

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
                        fontsize=5.5,
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
