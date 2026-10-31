"""实验 9-4 让程序跑得更快：测速度、比较输入尺寸、导出 ONNX 格式。

在 labs/ch09 目录下运行：python lab9_4_speed.py
"""
# %% 第 1 步：写一个测速函数——先热身，再跑 20 次取平均
import time

import cv2
from ultralytics import YOLO

img = cv2.imread("../data/images/drive_city.jpg")


def speed(model, imgsz=640, n=20):
    for _ in range(3):                                   # 热身：前几次比较慢，不算
        model(img, imgsz=imgsz, verbose=False)
    t0 = time.perf_counter()
    for _ in range(n):
        r = model(img, imgsz=imgsz, verbose=False)[0]
    ms = (time.perf_counter() - t0) / n * 1000
    return ms, r


# %% 第 2 步：同一个模型，输入尺寸 640、480、320 比一比
det = YOLO("yolo26n.pt")
print(f"{'输入尺寸':<6}{'每张毫秒':>8}{'每秒张数':>8}{'找到目标':>8}")
for size in (640, 480, 320):                             # ← 改这里，试试 256、800
    ms, r = speed(det, size)
    print(f"{size:<10}{ms:>10.1f}{1000 / ms:>10.1f}{len(r.boxes):>10}")

# %% 第 3 步：整个助手（检测 + 分割 + 深度）一张图要多久
sem, depth = YOLO("yolo26n-sem.pt"), YOLO("yolo26n-depth.pt")
parts = {"检测": det, "分割": sem, "深度": depth}
times = {k: speed(m)[0] for k, m in parts.items()}
for k, t in times.items():
    print(f"{k}：{t:.0f} 毫秒")
total = sum(times.values())
print(f"三个模型合计 {total:.0f} 毫秒/张，约 {1000 / total:.1f} 张/秒")

# %% 第 4 步：导出 ONNX 格式，再测一次（需要联网，第一次会自动安装 onnx，较慢）
try:
    det.export(format="onnx", imgsz=640)                 # 生成 yolo26n.onnx
    onnx_det = YOLO("yolo26n.onnx")
    ms, r = speed(onnx_det, 640)
    print(f"ONNX 版：{ms:.1f} 毫秒/张，找到 {len(r.boxes)} 个目标")
except Exception as e:                                   # 没装好也不影响前面的结果
    print("导出 ONNX 没成功：", e)
