# DRL - Wind-aware Fixed-wing UAV Routing

本项目提供了一个基于强化学习的完整流程，使用 **JSBSim** 作为动力学仿真器，让固定翼无人机在存在空间变化风场的情况下，从起点 A 规划到终点 B 的低风速航线。推理阶段可选配 **FlightGear** 实时可视化，便于对策略进行肉眼验证。针对你的硬件（i7-14650HX + RTX4060 Laptop，CUDA 12.8），训练默认使用 `stable-baselines3` 的 PPO 算法并依据配置选择 CPU/GPU。  

## 项目结构

```
DRL/
├── requirements.txt                 # Python 依赖
├── src/rl_wind_uav/
│   ├── __init__.py                  # 包导出
│   ├── train.py                     # 训练入口
│   ├── inference.py                 # 推理/可视化入口
│   ├── callbacks.py                 # PyCharm/终端友好的训练回调
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

## 快速开始（PyCharm 或命令行）

1. **克隆与依赖安装**：完成上文依赖安装后，将 `config/train.json` 与 `config/inference.json` 中的 `"/path/to/JSBSim"` 修改为真实的 JSBSim 目录。
2. **在 PyCharm 中运行**：
   - `train.py` 与 `inference.py` 可直接右键运行；脚本会自动读取同名 JSON 配置。
   - 如需指定 FlightGear，可在 `config/inference.json` 中把 `"env.enable_flightgear"` 设为 `true` 并填写 `"env.flightgear_path"`（或留空并保证 `fgfs` 在 PATH 中）。
   - 也可以在 *Run Configuration → Environment variables* 中添加 `JSBSIM_ROOT=/your/jsbsim/root`、`FLIGHTGEAR_EXE=/path/to/fgfs`，用于覆盖配置文件。
3. **命令行运行**：可使用下文的训练/推理示例命令，CLI 参数优先级最高，可覆盖 JSON 与环境变量。

训练期间会在终端展示 `tqdm` 进度条，并且 `ConsoleLogCallback` 会定期在 PyCharm/终端输出最近完成的若干回合平均回报、最高回报、执行
速度等信息，便于实时观察训练效果。你也可以在 `runs/ppo/tb` 中打开 TensorBoard 观察详细曲线。默认每 100000 步在 `runs/ppo/checkpoints/`
下保存一份策略快照，最终权重则写入 `runs/ppo/ppo_wind_route_final.zip`，方便在训练过程中随时切换或回滚模型。

## 配置文件结构概览

`config/train.json` 与 `config/inference.json` 集中管理训练/推理需要的大部分参数，核心字段说明如下：

- `jsbsim_root`：指向本地 JSBSim 安装的根目录，必须包含 `aircraft/` 等子目录。
- `device`：Torch 设备字符串（如 `cpu`、`cuda`、`cuda:0`），也可设置为 `auto` 交由 PyTorch 自动判断。
- `env`：环境参数集合，对应 `RouteConfig`；可以修改初始/目标经纬度、飞行时长、动作保持时间、姿态限制，或在推理时启用 FlightGear：
  - `enable_flightgear` / `flightgear_path` / `flightgear_port` / `run_fg_headless` 控制可视化。
  - `suppress_jsbsim_output` 为 `true` 时会在训练/推理初始化阶段静默 JSBSim 的大量机型参数打印；若需要调试底层 FDM，可改为 `false` 以查看原始输出。
  - `wind` 子段映射到 `WindFieldConfig`，可配置风场随机种子、网格尺度、最大风速和高度层数。
- `checkpoint`（训练）：设置 `freq`（保存步数间隔，0 表示关闭）与 `prefix`（保存文件名前缀）。
- `console_log`（训练）：`interval_steps` 控制每隔多少环境步在 PyCharm/终端打印一条训练摘要，设为 0 可关闭。
- `ppo`（训练）：覆盖 PPO 算法超参，如学习率、`n_steps`、`batch_size`、`clip_range`、`policy_kwargs` 等；未填写则采用 Stable-Baselines3 默认值。
- `deterministic`（推理）：`true` 表示使用确定性动作，`false` 为随机采样策略输出。
- 其他顶层字段如 `total_steps`、`eval_freq`、`logdir`、`episodes`、`model` 等分别控制训练步数、评估频率、日志目录、推理轮次数和策略权重位置。

命令行参数始终具有最高优先级，可在临时实验时覆盖 JSON 中的设置；环境变量（例如 `JSBSIM_ROOT`、`RL_WIND_UAV_DEVICE`）次之。

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
- 默认 `device` 取自配置（默认为 `auto`），如安装了 CUDA 12.8 对应的 PyTorch 会自动使用 RTX4060 Laptop GPU；也可以在命令行传入 `--device cuda:0`、修改 `config/train.json` 的 `"device"` 字段，或设置环境变量 `RL_WIND_UAV_DEVICE` 来强制选择 CPU/GPU。
- 通过命令行 `--checkpoint-freq 50000` 与 `--checkpoint-prefix custom_name`，或直接编辑配置中的 `checkpoint.freq` / `checkpoint.prefix`，可以调整权重定期保存行为；`--console-log-interval 20000` 或 `console_log.interval_steps` 可控制 PyCharm 控制台摘要输出频率；`--no-progress-bar` 可在服务器或
  无交互终端下关闭进度条输出。

### 无需命令行参数（PyCharm/IDE 运行）

如果直接在 PyCharm 等 IDE 中点击运行脚本，默认不会传入命令行参数。项目提供了 `config/train.json` 与 `config/inference.json`
两个配置文件，脚本会自动加载其中的参数：

1. 将 `config/train.json` 与 `config/inference.json` 中的 `"/path/to/JSBSim"` 修改为本机的 JSBSim 目录。
2. 如需更改训练超参数、checkpoint 频率（`checkpoint.freq`）、模型命名前缀（`checkpoint.prefix`）、是否显示进度条（`progress_bar`）、
   PyCharm 控制台输出频率（`console_log.interval_steps`）、
   PPO 学习率（`ppo.learning_rate`）、风场参数（`env.wind.*`）、姿态限制（`env.max_bank_deg` / `env.max_pitch_deg`）或推理轮次数、
   默认模型路径、训练/推理所用设备（`device`）等，也可直接编辑对应键值。
3. 要启用 FlightGear，可在 `config/inference.json` 的 `env.enable_flightgear` 设置为 `true` 并填写 `env.flightgear_path`，或在环境变量/CLI 中指定。
4. 仍可在 IDE 的 Run Configuration 中覆盖任意参数；命令行参数优先级高于配置文件。
5. 也支持通过设置环境变量 `JSBSIM_ROOT`（以及可选的 `FLIGHTGEAR_EXE`）来指定路径。
6. 想临时覆盖配置中的 `device` 时，可在运行前设置环境变量 `RL_WIND_UAV_DEVICE`（例如 `cpu`、`cuda:0`）。

当配置文件或环境变量缺失必填参数时，脚本会给出友好的错误提示，帮助快速定位问题。

## 推理与 FlightGear 可视化

```bash
python -m rl_wind_uav.inference \
  --model runs/ppo/ppo_wind_route_final.zip \
  --jsbsim-root /path/to/JSBSim \
  --episodes 3 \
  --flightgear \
  --flightgear-path /usr/games/fgfs
```

- 启用 `--flightgear` 后，环境会尝试启动 FlightGear 并通过 UDP 端口 (默认 5502) 进行 FDM 数据交换。若未在 PATH 中找到 `fgfs` 可通过 `--flightgear-path` 显式指定。
- 推理阶段同样遵循 `device` 配置：可在命令行传入 `--device cpu`/`--device cuda`，或在 `config/inference.json` 及环境变量 `RL_WIND_UAV_DEVICE` 中设定所需设备。
- 推理脚本会在终端输出每一步的奖励、与目标的相对位置及局部风向，便于离线分析。你也可以通过 `--model runs/ppo/checkpoints/ppo_wind_route_500000_steps.zip`
  等参数切换到某次中间 checkpoint，从而对比不同阶段策略的效果。

## 与 FlightGear 联动注意事项

1. 第一次启动 FlightGear 建议先手动运行一次以生成配置文件，并在设置中开启 *External FDM*（或直接使用脚本提供的 `--fdm=external` 参数）。
2. 如果希望更精细的可视化，可将 `RouteConfig` 中的 `run_fg_headless` 设置为 `False`，以窗口模式运行 FlightGear。
3. FlightGear 与 JSBSim 同时运行时资源消耗较大，建议在推理（非训练）阶段使用。

## 常见问题

- **找不到 `fgfs`**：设置环境变量 `FLIGHTGEAR_EXE`、在命令行使用 `--flightgear-path`，或在 `config/inference.json` 中填写 `"env.flightgear_path"`。
- **风场尺度调整**：通过 `WindFieldConfig(max_speed=..., scale=..., grid_size=...)` 控制风强与空间平滑度，训练前可结合真实风场数据进行初始化。
- **策略震荡/无法收敛**：尝试增大 `goal_threshold_m`、调整奖励系数或缩短动作保持时间；也可引入 `VecNormalize`、学习率调节等稳定技巧。
- **推理时仍看到 JSBSim 大量飞机参数输出**：确认 `config/inference.json`（或训练时的 `config/train.json`）中的 `env.suppress_jsbsim_output` 为 `true`。如需排查 JSBSim 配置，可暂时将其设为 `false` 以恢复详细输出。
- **Windows 下在 PyCharm 运行时报 `numpy._core.multiarray` 无法导入**：某些 IDE 在启动 Conda 解释器时不会自动注入 `Library/bin` 到 `PATH`，导致 MKL/BLAS DLL 无法找到。训练与推理脚本现已自动补充该目录；若仍报错，请确认使用的是 Conda 虚拟环境，并在 `Run Configuration → Environment variables` 中设置 `CONDA_PREFIX`（或手动将 `<conda_env>\Library\bin` 加入 `PATH`）。

## 许可

本仓库示例代码以 MIT 许可证发布，可按需修改拓展。
