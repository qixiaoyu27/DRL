# DRL 自定义点对点飞行环境

本示例基于 [gym-jsbsim](https://github.com/Gor-Ren/gym-jsbsim) 的思路，构建了一个更简洁的点对点飞行强化学习环境。飞机需要在有风背景下从 A 点安全飞往 B 点，并避开两枚圆形障碍物。脚本同时集成了 **PPO、SAC、TD3** 三种算法，默认使用 CUDA 12.8 + RTX 4060 Ti 8G 进行训练，并在训练完成后生成奖励-回合曲线。

## 环境要点

- **状态**：包含坐标、航向、速度、相对目标角度、风向以及最近障碍距离等 11 个归一化特征。
- **动作**：二维连续控制（油门、转向率）。
- **风场**：3 级及以下的随机风速向量，方向均匀采样。
- **奖励函数**：以向目标的前进距离为主，配合时间惩罚、靠近/撞击障碍惩罚、越界惩罚以及成功奖励。

## 快速开始

1. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

2. 使用指定算法训练（示例：PPO）：

   ```bash
   python train_rl.py --algo ppo --timesteps 500000 --device cuda
   ```

   - `--algo`：`ppo` / `sac` / `td3`
   - `--timesteps`：训练步数，可根据显存或时长调整。
   - `--device`：`cuda` 时默认占用第一块 GPU（建议 CUDA 12.8 + RTX 4060 Ti 8G）。

3. 训练结束后，会在 `outputs/` 下获得：

   - `reward_curve.png`：奖励-回合图表。
   - `{algo}_policy.zip`：保存的策略权重。

## 目录结构

```
DRL/
├── envs/
│   ├── __init__.py
│   └── point_to_point_env.py   # 自定义环境，含详细中文注释
├── train_rl.py                 # 训练入口，可选 PPO/SAC/TD3
├── requirements.txt           # 依赖清单
└── README.md
```

## 备注

- 如需更复杂的飞行力学，可直接接入原生 `gym-jsbsim` 环境并复用本训练脚本。
- 若希望可视化飞行轨迹，可在 `PointToPointAvoidEnv.render` 中自行扩展绘图逻辑。
