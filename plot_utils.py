"""
plot_utils.py — Shared constants, style, and utilities for CAS experiment figures.

Imported by both gen_plots_old.py and gen_plots.py.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import lmdb
import msgpack
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

# ── Default paths ──────────────────────────────────────────────────────────────
_script_dir = Path(__file__).resolve().parent

from dotenv import load_dotenv

load_dotenv()

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


def setup_style() -> None:
    """Configure matplotlib for publication-quality academic figures."""
    try:
        import scienceplots  # type: ignore[import]  # noqa: F401

        plt.style.use(["science", "no-latex"])
        return
    except ImportError:
        pass

    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    serif = next((f for f in _SERIF_PREF if f in available), "DejaVu Serif")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [serif],
            "mathtext.fontset": "stix",
            "font.size": 15,
            "axes.labelsize": 15,
            "axes.titlesize": 17,
            "axes.titlepad": 6,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
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
    "structure_m9": "Structure",
    "opinion_m9": "Opinion",
}

AQ_ORDER = ["a_gt_q", "q_gt_a"]
AQ_LABEL = {"a_gt_q": r"$\alpha > q$", "q_gt_a": r"$q > \alpha$"}

REPOST_ORDER = ["p25", "p0"]
REPOST_LABEL = {"p25": r"$p = 0.25$", "p0": r"$p = 0.00$"}

# Standard per-metric display info used across overview plots
METRICS: List[Tuple[str, str]] = [
    ("log_convergence_time", r"Log-convergence time $\log(1+t^*)$"),
    ("final_variance", "Final opinion variance"),
    ("final_magnetization", r"Final magnetization $|\bar{x}|$"),
    ("community_count", "Community count"),
]

# ── Okabe–Ito colorblind-friendly palette ─────────────────────────────────────
_OI = [
    "#E69F00",  # 0 orange
    "#56B4E9",  # 1 sky blue
    "#009E73",  # 2 bluish green
    "#F0E442",  # 3 yellow
    "#0072B2",  # 4 blue
    "#D55E00",  # 5 vermillion
    "#CC79A7",  # 6 reddish purple
    "#999999",  # 7 grey
]

RECSYS_COLOR = {
    "random": _OI[1],  # sky blue
    "structure_m9": _OI[5],  # vermillion
    "opinion_m9": _OI[2],  # bluish green
}

AQ_COLOR = {"a_gt_q": _OI[0], "q_gt_a": _OI[4]}
REPOST_COLOR = {"p25": _OI[5], "p0": _OI[7]}

# Markers for condition encoding in scatter plots
AQ_MARKER = {"a_gt_q": "o", "q_gt_a": "s"}  # circle / square
REPOST_ALPHA = {"p25": 0.90, "p0": 0.45}  # filled / faded

# ── LMDB record key pattern ────────────────────────────────────────────────────
_NAME_RE = re.compile(
    r"^(?P<dynamics>hk|deffuant|galam|voter)"
    r"-(?P<recsys>random|structure_m9|opinion_m9)"
    r"-(?P<aq>a_gt_q|q_gt_a)"
    r"-(?P<repost>p\d+)"
    r"-r(?P<rep>\d+)$"
)

# All numeric fields persisted in LMDB
_ALL_NUMERIC_FIELDS = [
    "log_convergence_time",
    "final_variance",
    "final_magnetization",
    "community_count",
    "n_closed_triangles",
    "opinion_peak_count",
    "modularity",
]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_dataframe(db_path: str) -> pd.DataFrame:
    """Load all LMDB stats records into a tidy pandas DataFrame.

    Includes the full field set:
        log_convergence_time, final_variance, final_magnetization,
        community_count, n_closed_triangles, opinion_peak_count, modularity.
    """
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
            for field in _ALL_NUMERIC_FIELDS:
                row[field] = data.get(field, np.nan)
            rows.append(row)
    env.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["rep"] = df["rep"].astype(int)
    for field in _ALL_NUMERIC_FIELDS:
        df[field] = pd.to_numeric(df[field], errors="coerce")
    return df


# ── Small helpers ──────────────────────────────────────────────────────────────
def mean_sem(series: pd.Series) -> Tuple[float, float]:
    """Return (mean, SEM) for a series, ignoring NaN."""
    vals = series.dropna()
    if len(vals) == 0:
        return np.nan, 0.0
    return float(vals.mean()), float(vals.sem())  # type: ignore[arg-type]


def save_fig(fig: Figure, out_dir: Path, name: str, fmt: str) -> None:
    """Save figure to *out_dir*/<name>.<fmt> and close it."""
    out = out_dir / f"{name}.{fmt}"
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)
