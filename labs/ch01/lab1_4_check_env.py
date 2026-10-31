# 实验 1-4 在自己电脑上安装环境：环境自检
# %% 运行后对照附录 C 检查每一项
import platform
import sys

print("Python 版本：", sys.version.split()[0], " 操作系统：", platform.system())
for name in ["cv2", "numpy", "matplotlib", "ultralytics", "torch"]:
    try:
        mod = __import__(name)
        print(f"  {name:12s} 已安装，版本 {mod.__version__}")
    except ImportError:
        print(f"  {name:12s} 没有安装 → 运行 pip install -r requirements.txt")

import torch
print("有没有 NVIDIA 显卡（GPU）：", "有，" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "没有，用 CPU 也能完成本书全部必做实验")
