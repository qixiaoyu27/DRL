# DRL - Wind-aware Fixed-wing UAV Routing

本项目提供了一个基于强化学习的完整流程，使用 **JSBSim** 作为动力学仿真器，让固定翼无人机在存在空间变化风场的情况下，从起点 A 规划到终点 B 的低风速航线。推理阶段可选配 **FlightGear** 实时可视化，便于对策略进行肉眼验证。针对你的硬件（i7-14650HX + RTX4060 Laptop，CUDA 12.8），训练默认使用 `stable-baselines3` 的 PPO 算法并自动选择 GPU。  

## 项目结构

```
DRL/
├── requirements.txt                 # Python 依赖
├── src/rl_wind_uav/
│   ├── __init__.py                  # 包导出
│   ├── train.py                     # 训练入口
│   ├── inference.py                 # 推理/可视化入口
│   └── env/
│       ├── __init__.py
│       ├── wind_field.py            # 程序化风场生成
│       └── fixed_wing_route_env.py  # JSBSim + Gymnasium 环境
└── README.md
```

## 环境依赖

1. **Python 3.10+**（推荐使用 Conda 或 venv 虚拟环境）。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. **JSBSim** 数据：确保本地有一份包含 `aircraft/`、`engine/` 等目录的 JSBSim 安装或源码仓库。训练与推理时需通过 `--jsbsim-root` 指定该路径。
4. 可选：安装 **FlightGear**（`fgfs`）用于推理可视化；Linux 下通常可通过包管理器安装，Windows / macOS 请从 [官网](https://www.flightgear.org/) 下载。

## 自定义风场

`src/rl_wind_uav/env/wind_field.py` 提供了 `WindField` 和 `WindFieldConfig`，可以通过随机种子、网格大小、层数、最大风速等参数控制风场形态。环境在 `reset()` 时会调用 `wind_field.reset(seed)` 以保证训练随机性。

## 强化学习环境说明

- **状态空间**：当前航向与目标方位夹角、航迹角、相对高度、空速、当前点风矢量、到目标的 ENU 分量。
- **动作空间**：二维连续向量 \([-1, 1]^2\)，分别线性映射到最大±25°的滚转指令与±10°俯仰指令（可在 `RouteConfig` 中调整）。
- **奖励函数**：鼓励靠近目标、保持高度、远离大风区，到达目标后额外奖励，过早触地则惩罚。
- **仿真步长**：默认 0.1 s，动作保持 0.5 s。所有参数均可在 `RouteConfig` 中自定义。

## 训练流程

```bash
python -m rl_wind_uav.train \
  --jsbsim-root /path/to/JSBSim \
  --logdir runs/ppo \
  --total-steps 4000000 \
  --num-envs 4 \
  --eval-freq 100000
```

- `--num-envs` 建议设置为 CPU 物理核心数的一半到全部；在 i7-14650HX 上可尝试 8。
- 训练日志与模型会输出到 `--logdir`，并写入 TensorBoard 日志，可使用 `tensorboard --logdir runs/ppo/tb` 观察学习曲线。
- 由于使用了 `device="auto"`，如安装了 CUDA 12.8 对应的 PyTorch，PPO 将自动使用 RTX4060 Laptop GPU。

### 无需命令行参数（PyCharm/IDE 运行）

如果直接在 PyCharm 等 IDE 中点击运行脚本，默认不会传入命令行参数。项目提供了 `config/train.json` 与 `config/inference.json`
两个配置文件，脚本会自动加载其中的参数：

1. 将 `config/train.json` 与 `config/inference.json` 中的 `"/path/to/JSBSim"` 修改为本机的 JSBSim 目录。
2. 如需更改训练超参数或默认的模型路径，也可直接编辑对应键值。
3. 仍可在 IDE 的 Run Configuration 中覆盖任意参数；命令行参数优先级高于配置文件。
4. 也支持通过设置环境变量 `JSBSIM_ROOT`（以及可选的 `FLIGHTGEAR_EXE`）来指定路径。

当配置文件或环境变量缺失必填参数时，脚本会给出友好的错误提示，帮助快速定位问题。

## 推理与 FlightGear 可视化

```bash
python -m rl_wind_uav.inference \
  --model runs/ppo/ppo_wind_route.zip \
  --jsbsim-root /path/to/JSBSim \
  --episodes 3 \
  --flightgear \
  --flightgear-path /usr/games/fgfs
```

- 启用 `--flightgear` 后，环境会尝试启动 FlightGear 并通过 UDP 端口 (默认 5502) 进行 FDM 数据交换。若未在 PATH 中找到 `fgfs` 可通过 `--flightgear-path` 显式指定。
- 推理脚本会在终端输出每一步的奖励、与目标的相对位置及局部风向，便于离线分析。

## 与 FlightGear 联动注意事项

1. 第一次启动 FlightGear 建议先手动运行一次以生成配置文件，并在设置中开启 *External FDM*（或直接使用脚本提供的 `--fdm=external` 参数）。
2. 如果希望更精细的可视化，可将 `RouteConfig` 中的 `run_fg_headless` 设置为 `False`，以窗口模式运行 FlightGear。
3. FlightGear 与 JSBSim 同时运行时资源消耗较大，建议在推理（非训练）阶段使用。

## 常见问题

- **找不到 `fgfs`**：设置环境变量 `FLIGHTGEAR_EXE` 或在命令行使用 `--flightgear-path` 指定。
- **风场尺度调整**：通过 `WindFieldConfig(max_speed=..., scale=..., grid_size=...)` 控制风强与空间平滑度，训练前可结合真实风场数据进行初始化。
- **策略震荡/无法收敛**：尝试增大 `goal_threshold_m`、调整奖励系数或缩短动作保持时间；也可引入 `VecNormalize`、学习率调节等稳定技巧。

## 许可

本仓库示例代码以 MIT 许可证发布，可按需修改拓展。
