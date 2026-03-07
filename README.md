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

## TODO（待实现的指标，在 run_experiments.py 中已注释）

- 平均对数收敛时间
- 最终意见簇数量（Leiden 社区检测）
- 方差 / 磁化率
- 闭合三角形数量
