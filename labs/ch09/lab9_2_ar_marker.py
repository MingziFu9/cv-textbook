"""实验 9-2 增强现实小实验：认出路面上的标记码，把一个导航箭头“贴”到路面上。

在 labs/ch09 目录下运行：python lab9_2_ar_marker.py
"""
# %% 第 1 步：生成一个标记码（ArUco 码），可以打印出来或显示在手机上
import cv2
import numpy as np

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)   # 4×4 格子的码，共 50 种
marker = aruco.generateImageMarker(dictionary, 7, 240)          # 7 号码，240×240 像素  ← 改这里
marker = cv2.copyMakeBorder(marker, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)  # 加白边才认得出
cv2.imwrite("lab9_2_marker.png", marker)
print("标记码大小：", marker.shape)

# %% 第 2 步：在照片里找标记码
img = cv2.imread("../data/images/marker_road.jpg")              # 路面上平放着一块标记码  ← 改这里
detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
corners, ids, rejected = detector.detectMarkers(img)
print("找到的码：", None if ids is None else ids.ravel().tolist())
quad = corners[0].reshape(4, 2)                                  # 四个角：左上、右上、右下、左下
print("四个角的像素坐标：\n", quad.round(1))
shown = img.copy()
aruco.drawDetectedMarkers(shown, corners, ids)
cv2.imwrite("lab9_2_detected.jpg", shown)

# %% 第 3 步：在“标记码自己的坐标”里画一个导航箭头（单位：标记码边长 = 240）
canvas = np.zeros((1000, 480, 3), np.uint8)                       # 一块纸，标记码在纸的底部中间
ox, oy = 120, 760                                                 # 标记码左上角在这块纸上的位置
arrow = np.array([[200, 990], [280, 990], [280, 300], [380, 300], [240, 80], [100, 300], [200, 300]])
cv2.fillPoly(canvas, [arrow], (0, 220, 255))                      # 黄色直行箭头，从标记码一直往前指
src = np.float32([[ox, oy], [ox + 240, oy], [ox + 240, oy + 240], [ox, oy + 240]])
H = cv2.getPerspectiveTransform(src, quad.astype(np.float32))     # 单应性矩阵：纸上的点 → 照片上的点
np.set_printoptions(suppress=True, precision=3)                    # 不用科学计数法显示
print("单应性矩阵 H：\n", H)

# %% 第 4 步：把箭头按 H 变形后贴到照片上
h, w = img.shape[:2]
warped = cv2.warpPerspective(canvas, H, (w, h))
mask = warped.sum(axis=2) > 0
ar = img.copy()
ar[mask] = (0.4 * img[mask] + 0.6 * warped[mask]).astype(np.uint8)   # 半透明，路面还能看见
print("箭头盖住的像素数：", int(mask.sum()))
cv2.imwrite("lab9_2_ar.jpg", ar)

# %% 第 5 步：对比——不做透视，只按外接矩形缩放、平移（像把箭头“立”在路上）
x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
sx, sy = bw / 240, bh / 240                                       # 横竖各缩放多少
M = np.float32([[sx, 0, x - ox * sx], [0, sy, y - oy * sy]])      # 只缩放、平移，没有透视
flat_arrow = cv2.warpAffine(canvas, M, (w, h))
m = flat_arrow.sum(axis=2) > 0
flat = img.copy()
flat[m] = (0.4 * img[m] + 0.6 * flat_arrow[m]).astype(np.uint8)
cv2.imwrite("lab9_2_flat.jpg", flat)
