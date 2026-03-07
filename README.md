# smp-experiment-cas

## 实验场景生成（scenarios.py）

完整的参数空间，总计 **800 次模拟**：

| 维度 | 连续模型 (HK / Deffuant) | 离散模型 (Galam / Voter) |
|------|--------------------------|--------------------------|
| 推荐系统 | Random、StructureM9、OpinionM9 | Random、StructureM9 |
| α vs q | α > q（Influence=0.5, Rate=0.05）<br>q > α（Influence=0.05, Rate=0.5） | 同左 |
| RepostRate p | p=0.25 / p=0.0 | 同左 |
| 重复次数 | 20 | 20 |

其他固定参数：500 节点、15 跟随数、Tolerance=0.45（连续模型）、5000 步上限。

## 主运行脚本（run_experiments.py）

```bash
python run_experiments.py              # 运行全部 800 次（默认并发 4）
python run_experiments.py --dry-run    # 仅打印统计，不运行
python run_experiments.py --concurrency 8 --max-step 2000
```

支持断点续跑（已完成的跳过），结果写入 `./run/<UniqueName>/`。

## 统计指标计算（stat_utils.py）

提供以下工具函数，可独立调用，也可通过 `compute_all_stats()` 一次性获取：

| 函数 | 返回值 | 说明 |
|------|--------|------|
| `get_triads_stats(A)` | `(n_triads, A_triads)` | 有向闭合三角形数量及矩阵 |
| `get_last_community_count(record)` | `(count, sizes_json)` | Leiden 社区数量及各社区大小（JSON） |
| `get_opinion_stats(record)` | `(variance, magnetization)` | 最终意见方差 / 绝对均值（磁化率） |
| `get_convergence_step(record)` | `int` | 意见收敛步数（最大单步变化首次低于 ε=1e-4） |
| `compute_all_stats(record)` | `dict` | 聚合所有指标，返回可 msgpack 序列化的字典 |

`compute_all_stats` 返回字段：

```
convergence_step        int     – 意见收敛步数
log_convergence_time    float   – log1p(convergence_step)
final_variance          float   – 最终步意见方差
final_magnetization     float   – 最终步 |mean opinion|（磁化率）
n_closed_triangles      int     – 最终图中有向闭合三角形数量
community_count         int     – Leiden 社区数量
community_sizes         str     – JSON {community_id: size}
```

## 统计批量采集（run_stats.py）

```bash
python run_stats.py                          # 处理全部已完成模拟（默认并发 4）
python run_stats.py --concurrency 8          # 指定并发数
python run_stats.py --db-path ./my.lmdb      # 指定输出数据库路径
python run_stats.py --force                  # 强制重算已存储条目
```

结果写入 LMDB 数据库（默认 `./run/stats.lmdb`）：

- **key**：场景 `UniqueName`（UTF-8 字节串）
- **value**：msgpack 编码的 `compute_all_stats` 字典
- 支持增量更新，已存储条目默认跳过

## 绘图（gen_plots.py）

| 文件名 | 内容 |
|---|---|
| `log_convergence_time.{fmt}` | 2×2 分组条形图（α/q 条件 × 转发率），按模型和推荐系统分组，含 ±1 SEM 误差棒 |
| `final_variance.{fmt}` | 同上，指标换为最终意见方差 |
| `final_magnetization.{fmt}` | 同上，指标换为最终磁化强度 |
| `community_count.{fmt}` | 同上，指标换为社区数量 |
| `recsys_delta.{fmt}` | Structure/Opinion 推荐系统相对 Random 基线的**相对变化（%）**，2×2 指标宫格 |
| `condition_comparison.{fmt}` | 两行：α>q vs q>α 条件对比（上）、转发率对比（下），每指标一列 |
| `summary_heatmap.{fmt}` | 列 z-score 归一化热图，行=（模型×推荐系统），列=（条件×转发率），标注原始均值 |

**字体优先级**：`scienceplots` → Times New Roman → Times → Liberation Serif → CMU Serif → DejaVu Serif（回退）。

运行方式：

```bash
python gen_plots.py                        # 默认 PDF，输出到 ./run/plots/
python gen_plots.py --format png           # PNG 格式
python gen_plots.py --db-path /path/to.lmdb --out-dir /path/to/out
```

## 环境变量

所有脚本均通过 `python-dotenv` 自动读取项目根目录下的 `.env` 文件。
命令行参数的优先级高于环境变量，环境变量的优先级高于代码内置默认值。

复制 `.env.example` 为 `.env` 并按需修改：

```bash
cp .env.example .env
```

| 环境变量 | 适用脚本 | 默认值 | 说明 |
|---|---|---|---|
| `SMP_BINARY_PATH` | `run_experiments.py` | `../social-media-models/smp` | Go 仿真二进制文件路径，支持 `~` 展开 |
| `SMP_BASE_PATH` | `run_experiments.py`<br>`run_stats.py` | `./run` | 仿真结果根目录，支持 `~` 展开 |
| `SMP_DB_PATH` | `run_stats.py`<br>`gen_plots.py` | `$SMP_BASE_PATH/stats.lmdb` | LMDB 统计数据库目录，支持 `~` 展开 |
| `SMP_PLOTS_PATH` | `gen_plots.py` | `$SMP_BASE_PATH/plots` | 图表输出目录，支持 `~` 展开 |

> **注意**：`SMP_DB_PATH` 的代码内置默认值为 `./run/stats.lmdb`，不会动态跟随 `SMP_BASE_PATH` 变化；如需两者联动，请在 `.env` 中同时设置两个变量。
