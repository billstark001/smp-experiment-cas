# smp-experiment-cas

Experiment scripts for the [social-media-models](https://github.com/billstark001/social-media-models) simulation framework.

## Experiment Scenario Generation (scenarios.py)

Full parameter space, totalling **800 simulations**:

| Dimension | Continuous models (HK / Deffuant) | Discrete models (Galam / Voter) |
|------|--------------------------|--------------------------|
| Recommendation system | Random, StructureM9, OpinionM9 | Random, StructureM9 |
| α vs q | α > q (Influence=0.5, Rate=0.05)<br>q > α (Influence=0.05, Rate=0.5) | Same as left |
| RepostRate p | p=0.25 / p=0.0 | Same as left |
| Repetitions | 20 | 20 |

Other fixed parameters: 500 nodes, 15 follows, Tolerance=0.45 (continuous models), 5000 steps maximum.

## Main Run Script (run_experiments.py)

```bash
python run_experiments.py              # Run all 800 simulations (default concurrency 4)
python run_experiments.py --dry-run    # Print statistics only; do not run
python run_experiments.py --concurrency 8 --max-step 2000
```

Supports resuming from interruptions (completed runs are skipped). Results are written to `./run/<UniqueName>/`.

## Statistics Computation (stat_utils.py)

Provides the following utility functions, which can be called independently or all at once via `compute_all_stats()`:

| Function | Return value | Description |
|------|--------|------|
| `get_triads_stats(A)` | `(n_triads, A_triads)` | Number of directed closed triangles and their matrix |
| `get_last_community_count(record)` | `(count, sizes_json)` | Number of Leiden communities and their sizes (JSON) |
| `get_opinion_stats(record)` | `(variance, magnetization)` | Final opinion variance / absolute mean (magnetization) |
| `get_convergence_step(record)` | `int` | Opinion convergence step (first step where max per-step change drops below ε=1e-4) |
| `compute_all_stats(record)` | `dict` | Aggregate all metrics; returns a msgpack-serializable dict |

`compute_all_stats` return fields:

```
convergence_step        int     – opinion convergence step
log_convergence_time    float   – log1p(convergence_step)
final_variance          float   – opinion variance at the final step
final_magnetization     float   – |mean opinion| at the final step (magnetization)
n_closed_triangles      int     – directed closed triangles in the final graph
community_count         int     – number of Leiden communities
community_sizes         str     – JSON {community_id: size}
```

## Batch Statistics Collection (run_stats.py)

```bash
python run_stats.py                          # Process all completed simulations (default concurrency 4)
python run_stats.py --concurrency 8          # Specify concurrency
python run_stats.py --db-path ./my.lmdb      # Specify output database path
python run_stats.py --force                  # Force recomputation of stored entries
```

Results are written to an LMDB database (default `./run/stats.lmdb`):

- **key**: scenario `UniqueName` (UTF-8 byte string)
- **value**: msgpack-encoded `compute_all_stats` dict
- Supports incremental updates; stored entries are skipped by default

## Plotting (gen_plots.py)

| Filename | Content |
|---|---|
| `log_convergence_time.{fmt}` | 2×2 grouped bar chart (α/q condition × repost rate), grouped by model and recommendation system, with ±1 SEM error bars |
| `final_variance.{fmt}` | Same as above, metric changed to final opinion variance |
| `final_magnetization.{fmt}` | Same as above, metric changed to final magnetization |
| `community_count.{fmt}` | Same as above, metric changed to community count |
| `recsys_delta.{fmt}` | Structure/Opinion recommendation system **relative change (%)** vs Random baseline, 2×2 metric grid |
| `condition_comparison.{fmt}` | Two rows: α>q vs q>α condition comparison (top), repost rate comparison (bottom), one column per metric |
| `summary_heatmap.{fmt}` | Column z-score normalised heatmap, rows=(model×recommendation system), columns=(condition×repost rate), with raw means annotated |

**Font priority**: `scienceplots` → Times New Roman → Times → Liberation Serif → CMU Serif → DejaVu Serif (fallback).

Run:

```bash
python gen_plots.py                        # Default PDF, output to ./run/plots/
python gen_plots.py --format png           # PNG format
python gen_plots.py --db-path /path/to.lmdb --out-dir /path/to/out
```

## Environment Variables

All scripts automatically read the `.env` file at the project root via `python-dotenv`.
Command-line arguments take precedence over environment variables, which in turn take precedence over built-in code defaults.

Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
```

| Environment Variable | Applicable Scripts | Default | Description |
|---|---|---|---|
| `SMP_BINARY_PATH` | `run_experiments.py` | `../social-media-models/smp` | Path to the Go simulation binary; supports `~` expansion |
| `SMP_BASE_PATH` | `run_experiments.py`<br>`run_stats.py` | `./run` | Simulation results root directory; supports `~` expansion |
| `SMP_DB_PATH` | `run_stats.py`<br>`gen_plots.py` | `$SMP_BASE_PATH/stats.lmdb` | LMDB statistics database directory; supports `~` expansion |
| `SMP_PLOTS_PATH` | `gen_plots.py` | `$SMP_BASE_PATH/plots` | Plot output directory; supports `~` expansion |

> **Note**: The built-in default for `SMP_DB_PATH` is `./run/stats.lmdb` and does not dynamically follow `SMP_BASE_PATH`; to link them, set both variables in `.env`.
