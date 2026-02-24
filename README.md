# DRL: 基于 JSBSim 的固定翼无人机风场全覆盖路径规划

本项目实现硕士课题原型：在 4km×4km 区域中，固定翼无人机在三级以下背景风 + 4 个 Rankine 涡旋的风场中进行全覆盖路径规划，算法使用 **PPO + LSTM + Attention + Residual**。

## 已实现能力

- 使用 **JSBSim** 驱动固定翼飞行仿真（默认机型 `c172p`）。
- 训练环境参数写死在代码中（`drl_uav/config.py`），可在 IDE 中直接运行，无需命令行传参。
- 自动生成并保存 **100 个风场环境** 到 `outputs/scenarios_100.json`。
- 训练期间实时保存奖励曲线图 `reward_curve.png`。
- 按固定步数保存模型权重，并同步运行评估生成轨迹图。

## 代码结构

```text
.
├── drl_uav
│   ├── config.py             # 写死参数配置（IDE 直接运行）
│   ├── scenario_manager.py   # 生成/加载100个风场环境(JSON)
│   ├── wind_field.py         # 背景风 + Rankine涡旋
│   ├── env.py                # JSBSim + 覆盖任务环境
│   ├── models.py             # Attention+Residual + Recurrent policy
│   └── evaluate.py           # 评估与轨迹绘图
├── train.py                  # 训练入口（无CLI参数）
└── requirements.txt
```

## 运行

```bash
pip install -r requirements.txt
python train.py
```

## 输出

默认输出目录：`outputs/`

- `scenarios_100.json`：100 个固定风场场景。
- `reward_curve.png`：训练回合奖励曲线。
- `models/ppo_lstm_attn_res_step_*.zip`：周期性模型权重。
- `models/eval_traj_step_*.png`：周期性评估轨迹图。
- `models/ppo_lstm_attn_res_final.zip`：最终模型。
- `final_eval_trajectory.png`：最终轨迹图。

## 单独评估

```bash
python -m drl_uav.evaluate
```
