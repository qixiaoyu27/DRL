from __future__ import annotations

"""批量执行消融实验（IDE内运行）。"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "drl_uav" / "training" / "train_ppo_residual.py"

# 这里示例化配置；如需并行可在IDE开启多个运行配置。
ABLATIONS = [
    {"name": "full", "USE_LSTM": "True", "USE_ATTENTION": "True", "USE_RESIDUAL_POLICY": "True"},
    {"name": "no_attn", "USE_LSTM": "True", "USE_ATTENTION": "False", "USE_RESIDUAL_POLICY": "True"},
    {"name": "no_residual", "USE_LSTM": "True", "USE_ATTENTION": "True", "USE_RESIDUAL_POLICY": "False"},
    {"name": "ppo_only", "USE_LSTM": "False", "USE_ATTENTION": "False", "USE_RESIDUAL_POLICY": "False"},
]


def main():
    for cfg in ABLATIONS:
        print(f"===== Running {cfg['name']} =====")
        code = TRAIN_SCRIPT.read_text(encoding="utf-8")
        for k in ["USE_LSTM", "USE_ATTENTION", "USE_RESIDUAL_POLICY"]:
            code = code.replace(f"{k} = True", f"{k} = {cfg[k]}")
            code = code.replace(f"{k} = False", f"{k} = {cfg[k]}")
        tmp = ROOT / "_tmp_train_ablation.py"
        tmp.write_text(code, encoding="utf-8")
        subprocess.run([sys.executable, str(tmp)], check=True)
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
