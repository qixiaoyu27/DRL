# Fixed-Wing UAV Route Planning with JSBSim and ERA5

本项目实现了一个用于固定翼无人机在 3D 空间中规划鲁棒航路的深度强化学习框架。系统综合使用 JSBSim 仿真器、ERA5 风场数据以及 PPO/SAC 算法，能够在保证快速到达目标的同时兼顾能耗，并在 3 级（约 10.8 m/s）以下风速条件下保持稳定。

## 功能概述

- ✅ **仿真环境**：基于 JSBSim 的固定翼无人机环境，融合 GPS、IMU、气压计读数。
- ✅ **风场建模**：支持加载 ERA5 NetCDF 数据，实时插值 3D 风矢量并驱动仿真。
- ✅ **强化学习算法**：提供 PPO 与 SAC 两种算法，可通过统一配置文件自由切换，默认使用 GPU (CUDA)。
- ✅ **训练可视化**：输出训练收敛曲线与风场热图中的飞行轨迹。
- ✅ **脚本化流程**：一键训练与评估脚本，便于在不同风场/航线场景下复用。

## 环境要求

- Python 3.9
- CUDA 12.8（可选，但建议以启用 GPU 加速）

### 依赖安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **注意**：`torch` 的 GPU 版本需根据本机 CUDA 版本手动选择，可参考 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)。

## ERA5 数据准备

1. 通过 Copernicus CDS 下载目标区域的 ERA5 风场（`u10`, `v10`, 可选 `w` 分量）。
2. 确保数据以 NetCDF (`.nc`) 格式存储，并包含纬度、经度、海拔（可选）与时间维度。
3. 在配置文件 `configs/default.yaml` 中的 `wind_field.era5_path` 指定数据路径。

## 训练

1. 编辑 `configs/default.yaml`（或复制成新的配置文件）以设定训练与环境参数。例如：

   ```yaml
   algorithm:
     name: sac
     total_timesteps: 300000
     device: cuda
   wind_field:
     era5_path: data/era5_wind.nc
   environment:
     initial_latitude: 34.5
     initial_longitude: -117.5
     target:
       latitude: 34.8
       longitude: -117.2
   output:
     model_dir: checkpoints
     plots_dir: plots
   ```

2. 启动训练：在 PyCharm 中直接运行 `train.py`（或在终端运行 `python train.py`）即可，脚本会自动读取 `configs/default.yaml`。

   - 若需要在不修改源码的情况下切换其他配置文件，可设置环境变量 `FIXEDWING_TRAIN_CONFIG=/path/to/custom.yaml` 后再运行。
   - 亦可在 Python 交互环境中调用 `train.main("configs/another.yaml")` 显式指定路径，方便在 Notebook/PyCharm 中复用。

训练完成后，将得到：

- `checkpoints/<algo>_model.zip`：保存的策略模型。
- `plots/training_curves.png`：奖励与损失曲线。
- `plots/wind_trajectory.png`：风场热图与飞行轨迹。

## 评估与热图生成

若需单独基于已训练模型生成风场热图，可直接运行 `scripts/generate_heatmap.py`。脚本会：

- 自动读取与训练相同的配置（可通过 `FIXEDWING_TRAIN_CONFIG` 或 `main(config_path=...)` 覆盖）。
- 默认加载 `checkpoints/<algo>_model.zip`，若需自定义模型路径，可在脚本顶部的 `MODEL_OVERRIDE` 中填入绝对/相对路径，或在调用 `main(model_path=...)` 时显式指定。

脚本会复用配置文件中的 ERA5、环境与可视化目录设置，输出图像默认保存至 `plots/wind_trajectory.png`。

## 代码结构

```text
fixedwing_rl/
├── envs/
│   └── fixedwing_jsbsim_env.py   # JSBSim 环境及配置
├── utils/
│   ├── callbacks.py              # 训练指标记录
│   ├── config.py                 # YAML 配置解析
│   ├── plotting.py               # 可视化工具
│   └── wind_field.py             # ERA5 风场读取与插值
configs/
└── default.yaml                  # 默认训练配置
scripts/
└── generate_heatmap.py           # 独立热图生成脚本
train.py                          # 训练入口
```

## 扩展建议

- 集成矢量化风场更新以提升大范围场景的查询效率。
- 添加多目标航点规划或任务级奖励函数。
- 接入更多传感器或故障模式以增强鲁棒性。

## 参考

- [DQN-uav (luzhixing12345)](https://github.com/luzhixing12345/DQN-uav)
- [JSBSim 飞行动力学模型](https://jsbsim-team.github.io/)
- [ERA5 气象数据](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
