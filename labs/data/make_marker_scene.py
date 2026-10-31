"""编者用：生成实验 9-2 的“测试场地路面上平放着一块标记码”的合成照片 images/marker_road.jpg。

在 labs/data 目录下运行：python make_marker_scene.py
"""
import cv2
import numpy as np

road = cv2.imread("images/drive_highway.jpg")
aruco = cv2.aruco
marker = aruco.generateImageMarker(aruco.getPredefinedDictionary(aruco.DICT_4X4_50), 7, 240)
card = cv2.copyMakeBorder(marker, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)   # 白边，像打印出来的纸
card = cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)
s = card.shape[0]
src = np.float32([[0, 0], [s, 0], [s, s], [0, s]])
dst = np.float32([[590, 555], [735, 555], [768, 640], [557, 640]])   # 平放在路面上，近大远小
H = cv2.getPerspectiveTransform(src, dst)
h, w = road.shape[:2]
warped = cv2.warpPerspective(card, H, (w, h))
mask = cv2.warpPerspective(np.full((s, s), 255, np.uint8), H, (w, h))
shade = (0.85 * warped).astype(np.uint8)                              # 户外纸面略暗一点
out = np.where(mask[..., None] > 0, shade, road)
out = cv2.GaussianBlur(out, (3, 3), 0)
noise = np.random.default_rng(0).normal(0, 4, out.shape)
out = np.clip(out + noise, 0, 255).astype(np.uint8)
cv2.imwrite("images/marker_road.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("已生成 images/marker_road.jpg", out.shape)
