"""实验 6-4 谁离我更近：用深度估计模型生成行车照片的深度图，结合检测框比较几辆车的远近。

在 labs/ch06 目录下运行：python lab6_4_depth.py
"""
# %% 第 1 步：深度估计，每个像素一个“远近”数值
import cv2
import numpy as np
from ultralytics import YOLO

img = cv2.imread("../data/images/drive_city.jpg")     # ← 改这里
depth_model = YOLO("yolo26n-depth.pt")                 # 单张照片估计远近的模型
r = depth_model(img, verbose=False)[0]
depth = r.depth.data.cpu().numpy()                     # 和原图一样大，数值越大越远（单位约为米）
print("深度图形状：", depth.shape)
print("最近", round(depth.min(), 1), "  最远", round(depth.max(), 1), "  平均", round(depth.mean(), 1))
cv2.imwrite("lab6_4_depth.jpg", r.plot())              # 近处偏红、远处偏蓝

# %% 第 2 步：自己把深度图涂成颜色
d = (depth - depth.min()) / (depth.max() - depth.min())   # 缩放到 0～1
gray = ((1 - d) * 255).astype(np.uint8)                # 反过来：近 → 亮，远 → 暗
color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)      # 亮 → 红，暗 → 蓝
cv2.imwrite("lab6_4_color.jpg", np.vstack([img, color]))   # 原图和彩色深度图上下拼在一起
h, w = depth.shape
for y in [h - 10, h // 2, h // 3]:                     # 看看画面中间一列从下往上的深度
    print(f"第 {y} 行中间的深度：{depth[y, w // 2]:.1f}")

# %% 第 3 步：找出画面中的车和人，取框里深度的中位数作为它的距离
det = YOLO("yolo26n.pt")(img, verbose=False)[0]
items = []
for b, c in zip(det.boxes.xyxy.int().tolist(), det.boxes.cls.tolist()):
    x1, y1, x2, y2 = b
    dist = float(np.median(depth[y1:y2, x1:x2]))       # 框里所有像素深度的中位数
    items.append((dist, det.names[int(c)], b))
items.sort()                                           # 按距离从近到远排序
for dist, name, b in items:
    print(f"{name:<8} 约 {dist:5.1f} 米  框 {b}")

# %% 第 4 步：在图上标出距离，最近的目标太近就提醒
out = img.copy()
for dist, name, (x1, y1, x2, y2) in items:
    col = (0, 0, 255) if dist < 10 else (0, 200, 0)    # 10 米以内画红框  ← 改这里
    cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
    cv2.putText(out, f"{dist:.0f}m", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
cv2.imwrite("lab6_4_near.jpg", out)
dist, name, _ = items[0]                               # 排在第一个的就是最近的
tip = "注意保持距离" if dist < 10 else "距离正常"
print(f"最近的是 {name}，约 {dist:.1f} 米 → {tip}")
