# DRL：固定翼无人机残差强化学习全覆盖路径规划（硕士课题模板）

本仓库给出一套**可直接在IDE中运行**的完整代码框架，满足你提出的要求：
- 4km×4km地图；
- 固定翼巡航25m/s，高度100m（二维平面建模，高度固定）；
- 背景风（3级及以下）+ 约4个Rankine涡旋风场；
- 基线方法（牛耕法）+ RL残差控制；
- SB3 PPO基线，提供 LSTM(RecurrentPPO) + 注意力特征提取改进；
- 定时输出奖励曲线、断点续训、消融开关；
- 独立风场设计器UI + 随机生成100训练环境脚本；
- 丰富评估脚本输出论文图/表。

---

## 1. 项目结构

```text
DRL/
├─ drl_uav/
│  ├─ envs/
│  │  ├─ wind_field.py                # Rankine涡旋与风场组合
│  │  └─ coverage_env.py              # 自定义理论环境（微分方程推进）
│  ├─ models/
│  │  └─ attention_extractor.py       # 注意力特征提取器
│  ├─ training/
│  │  ├─ callbacks.py                 # 奖励曲线定时绘图
│  │  └─ train_ppo_residual.py        # 训练脚本（含断点续训、消融开关）
│  ├─ tools/
│  │  ├─ wind_field_designer.py       # 独立风场设计器UI（PyQt5）
│  │  └─ generate_random_wind_cases.py# 生成100个随机风场
│  └─ evaluation/
│     ├─ evaluate_models.py           # 多风场评估与汇总表
│     ├─ plot_paper_figures.py        # 论文图表生成
│     ├─ ablation_runner.py           # 消融实验批量运行
│     └─ smoke_test_env.py            # 环境烟雾测试
└─ wind_cases/                        # 自动生成风场json（运行后出现）
```

---

## 2. 核心创新点（可写入论文）

1. **残差策略学习**：将传统牛耕法作为先验策略，PPO只学习修正量（Residual Action），显著降低探索难度。  
2. **风场前视感知建模**：智能体观测自身状态 + 前方120°/1km扇形区域风矢量采样。  
3. **PPO改进**：采用 `RecurrentPPO (LSTM)` 处理时序扰动，并引入 `Multi-Head Attention` 聚合前视风场token。  
4. **面向论文的数据输出闭环**：训练曲线、评估CSV、统计表、图像自动生成。

---

## 3. IDE运行顺序（无终端参数）

> 所有参数均写死在代码中，直接点击运行文件即可。

1. 运行 `drl_uav/tools/generate_random_wind_cases.py` 生成100个训练环境；
2. 可选运行 `drl_uav/tools/wind_field_designer.py` 人工设计风场并保存json；
3. 运行 `drl_uav/training/train_ppo_residual.py` 开始训练（自动断点续训）；
4. 运行 `drl_uav/evaluation/evaluate_models.py` 输出评估结果与表格；
5. 运行 `drl_uav/evaluation/plot_paper_figures.py` 生成论文图。

---

## 4. 关键参数说明（默认）

- 地图尺寸：`4000m`
- 巡航速度：`25m/s`
- 动力学步长：`1s`
- 相机FOV：`120°`
- 前视距离：`1000m`
- 并行环境：`N_ENVS = min(8, CPU核数)`（适配R5 5600X）
- 训练步数：`2,000,000`
- 设备：默认 `cuda`（RTX4060Ti）

---

## 5. 消融实验开关

编辑 `drl_uav/training/train_ppo_residual.py`：
- `USE_LSTM`
- `USE_ATTENTION`
- `USE_RESIDUAL_POLICY`

也可运行 `drl_uav/evaluation/ablation_runner.py` 自动批量执行示例配置。

---

## 6. 论文可直接引用的输出

在 `paper_outputs/` 中生成：
- `evaluation_metrics.csv`（逐风场明细）
- `evaluation_summary_table.csv`（均值/方差统计）
- `fig_coverage_boxplot.png`（覆盖率箱线图）
- `fig_steps_hist.png`（步数分布图）
- `fig_coverage_by_case.png`（跨场景覆盖率图）

---

## 7. 说明

- 本项目不依赖JSBSim物理仿真训练，而是采用**理论动力学微分方程环境**；
- 适合硕士论文场景：结构清晰，便于扩展复杂约束（转弯半径、过载上限、能耗项、禁飞区等）；
- 若你需要，我可以继续在此基础上补：
  - 实验章节模板（中英双语）；
  - 对比算法（A2C/SAC/TD3）统一评估脚本；
  - 显著性检验（t-test / Wilcoxon）与LaTeX表格自动导出。
