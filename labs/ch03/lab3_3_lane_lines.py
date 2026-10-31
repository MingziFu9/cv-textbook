# 实验 3-3 车道线检测：灰度 → 高斯去噪 → Canny → 感兴趣区域 → 霍夫直线
# %% 第 1 步：找边缘
import cv2
import numpy as np

img = cv2.imread("../data/images/lane_white.jpg")          # ← 改这里：换成 lane_yellow.jpg
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)                           # ← 改这里：低门槛、高门槛
print("边缘像素：", cv2.countNonZero(edges))

# %% 第 2 步：只留下梯形的路面区域
h, w = edges.shape
top = int(h * 0.6)                                         # 梯形上边：画面 60% 高度处  ← 改这里
trapezoid = np.array([[(int(w * 0.1), h), (int(w * 0.45), top), (int(w * 0.55), top), (int(w * 0.95), h)]])  # 左下、左上、右上、右下
roi_mask = np.zeros_like(edges)                            # 一张全黑的图
cv2.fillPoly(roi_mask, trapezoid, 255)                     # 把梯形涂白
roi_edges = cv2.bitwise_and(edges, roi_mask)               # 两张都是白的地方才留下
print("梯形里的边缘像素：", cv2.countNonZero(roi_edges))

# %% 第 3 步：霍夫变换找直线
lines = cv2.HoughLinesP(roi_edges, rho=2, theta=np.pi / 180, threshold=30,
                        minLineLength=20, maxLineGap=100)   # ← 改这里：threshold 越大要求票数越多
print("找到线段：", len(lines))

# %% 第 4 步：按斜率分成左、右两组，画回原图
show = img.copy()
left = right = 0
for x1, y1, x2, y2 in lines.reshape(-1, 4):
    k = (y2 - y1) / (x2 - x1 + 1e-6)                       # 斜率：往右下为正
    if k < -0.5: left += 1; color = (0, 0, 255)            # 左车道线：往右上斜，斜率为负
    elif k > 0.5: right += 1; color = (0, 255, 0)          # 右车道线：斜率为正
    else: continue                                         # 太平的线不要
    cv2.line(show, (x1, y1), (x2, y2), color, 4)
print(f"左边 {left} 条，右边 {right} 条")
cv2.imwrite("lab3_3_lanes.jpg", show)
