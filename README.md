# DRL Fixed-Wing UAV Area Scanning

本仓库提供一个可运行的研究级代码框架，用于复现你描述的硕士课题核心功能：
- **40×40 动态栅格**（4km²）区域扫描环境；
- **时变背景风 + Rankine 涡旋**复合扰动；
- **Residual PPO**（基准牛耕法 + 残差策略）；
- **LSTM 时序感知 + Self-Attention 空间特征抽取**；
- **分阶段课程学习**（无风 → 多气旋）。

> 说明：当前实现是“研究原型版”，可直接训练与评估，并输出 CSV 与图表。JSBSim 高保真接口可以在该框架上进一步替换/接入。

## 目录结构

- `drl_uav/core/environment.py`：风场、涡旋、区域扫描环境、牛耕法基准控制器。
- `drl_uav/algorithms/residual_ppo.py`：LSTM + Attention + Residual PPO 模型与训练器。
- `drl_uav/scripts/train.py`：课程学习训练脚本。
- `drl_uav/scripts/evaluate.py`：固定高难场景评估，输出轨迹图。
- `drl_uav/scripts/plot_results.py`：训练/评估指标可视化。
- `drl_uav/configs/default.yaml`：实验参数配置。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练

```bash
python -m drl_uav.scripts.train --config drl_uav/configs/default.yaml --out drl_uav/outputs/train
```

输出：
- `drl_uav/outputs/train/policy.pt`
- `drl_uav/outputs/train/train_log.csv`

## 评估

```bash
python -m drl_uav.scripts.evaluate --config drl_uav/configs/default.yaml --model drl_uav/outputs/train/policy.pt --out drl_uav/outputs/eval --episodes 5
```

输出：
- `drl_uav/outputs/eval/eval_metrics.csv`
- `drl_uav/outputs/eval/trajectory_ep*.png`

## 绘图与汇总

```bash
python -m drl_uav.scripts.plot_results --train-log drl_uav/outputs/train/train_log.csv --eval-log drl_uav/outputs/eval/eval_metrics.csv --out drl_uav/outputs/plots
```

输出：
- `drl_uav/outputs/plots/training_curves.png`
- `drl_uav/outputs/plots/eval_coverage.png`
- `drl_uav/outputs/plots/summary.csv`

## 指标建议

建议重点统计：
1. 覆盖率（最终值、95% 达成率）；
2. 轨迹平滑性（控制量变化率、航迹曲率）；
3. 抗风能力（侧风下航迹偏移、蟹形修正角）；
4. 样本效率（达到给定覆盖率阈值所需步数/更新次数）；
5. 泛化能力（未见风场参数下的性能下降幅度）。

## JSBSim 接入建议

当前环境动力学为轻量化近似。若要切换到 JSBSim：
1. 在 `FixedWingScanEnv.step()` 中替换状态推进为 JSBSim 一步积分；
2. 将动作映射到滚转/升降/油门目标，经 PID 转舵面；
3. 保留本项目的风场与覆盖率奖励结构；
4. 用相同训练脚本做对比实验（有/无残差、有/无课程）。
