"""编者用来生成 labs/data/images/ 中示例图片的脚本（学生不用运行）。

需要先把原始素材放在 RAW 目录：Udacity 行车视频、KITTI（Ultralytics 转换版）、Ultralytics medical-pills。
来源与许可见 labs/data/README.md。
"""
from pathlib import Path

import cv2
import numpy as np

RAW = Path("/home/claude/data-raw")          # 原始素材所在目录
OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)


def frame(video, i):
    cap = cv2.VideoCapture(str(RAW / video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, img = cap.read()
    return img


def save(name, img):
    cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(name, img.shape)


# 1. 行车照片
save("drive_highway.jpg", frame("project_video.mp4", 700))           # 高速公路，前方有车
save("lane_white.jpg", frame("solidWhiteRight.mp4", 100))
save("lane_yellow.jpg", frame("solidYellowLeft.mp4", 200))
city = cv2.imread(str(RAW / "kitti/images/train/005356.png"))         # 城市街道：车、行人、厢式车
save("drive_city.jpg", city)
save("street_crossing.jpg", cv2.imread(str(RAW / "kitti/images/train/004747.png")))  # 行人过马路

# 2. 模拟夜间画面：把城市街道调暗、偏蓝、加噪声（真实夜间照片的特点）
night = city.astype(np.float32) / 255
night = np.power(night, 2.6) * 0.55                                  # 整体变暗，暗部更暗
night[..., 0] *= 1.15                                                # 偏蓝
night += rng.normal(0, 0.02, night.shape)                            # 感光噪声
save("drive_night.jpg", np.clip(night * 255, 0, 255).astype(np.uint8))

# 3. 药片照片：编者用程序画的“桌面上的白色药片”（有大有小、有两颗挨在一起，用来练习轮廓计数）
pills = np.zeros((480, 640, 3), np.uint8)
pills[:] = (60, 95, 140)                                             # 棕色木纹桌面的底色（BGR）
for y in range(0, 480, 6):                                           # 简单的木纹
    cv2.line(pills, (0, y), (640, y + int(rng.integers(-4, 5))), (55, 88, 132), 2)
spots = [(90, 90, 28), (230, 70, 24), (400, 110, 30), (540, 80, 26), (130, 250, 26), (300, 240, 32),
         (356, 252, 26), (500, 260, 28), (80, 400, 30), (240, 390, 24), (420, 400, 28), (570, 390, 22)]
for x, y, rad in spots:                                              # 12 颗药片，其中 2 颗挨在一起
    cv2.circle(pills, (x + 3, y + 4), rad, (35, 55, 85), -1)        # 阴影
    cv2.circle(pills, (x, y), rad, (235, 238, 240), -1)              # 药片
    cv2.line(pills, (x - rad + 6, y), (x + rad - 6, y), (200, 205, 210), 2)   # 中间的刻痕
pills = cv2.GaussianBlur(pills, (5, 5), 0)
pills = np.clip(pills.astype(np.int16) + rng.normal(0, 6, pills.shape), 0, 255).astype(np.uint8)
save("pills.jpg", pills)

# 4. 证件照素材：编者用程序画的卡通头像（蓝底），不使用真人照片
por = np.zeros((480, 360, 3), np.uint8)
por[:] = (219, 142, 67)                                              # 证件照常用的蓝底（BGR）
cv2.ellipse(por, (180, 470), (150, 110), 0, 180, 360, (60, 60, 60), -1)      # 深色上衣
cv2.rectangle(por, (155, 300), (205, 370), (170, 200, 235), -1)              # 脖子
cv2.ellipse(por, (180, 220), (85, 105), 0, 0, 360, (180, 210, 245), -1)      # 脸
cv2.ellipse(por, (180, 165), (92, 70), 0, 180, 360, (30, 30, 40), -1)        # 头发
for ex in (148, 212):
    cv2.circle(por, (ex, 215), 8, (40, 40, 40), -1)                 # 眼睛
cv2.ellipse(por, (180, 270), (25, 10), 0, 0, 180, (60, 60, 160), 3)          # 微笑
save("portrait_blue.jpg", por)

# 5. 老照片：街景转成泛黄的黑白照，再加划痕和斑点
old = cv2.imread(str(RAW / "kitti/images/train/004139.png"))[:, 300:900]
gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY).astype(np.float32)
sepia = np.dstack([gray * 0.75, gray * 0.9, gray * 1.05])
sepia = cv2.GaussianBlur(sepia, (3, 3), 0)
scratch = np.zeros(gray.shape, np.uint8)
for _ in range(6):                                                  # 白色划痕
    x = int(rng.integers(0, gray.shape[1]))
    pts = np.array([[x + int(rng.integers(-30, 30)), 0], [x + int(rng.integers(-30, 30)), gray.shape[0]]])
    cv2.line(scratch, tuple(int(v) for v in pts[0]), tuple(int(v) for v in pts[1]), 255, 1)
for _ in range(40):                                                  # 白色斑点
    cv2.circle(scratch, (int(rng.integers(0, gray.shape[1])), int(rng.integers(0, gray.shape[0]))), int(rng.integers(1, 4)), 255, -1)
sepia[scratch > 0] = 235
save("old_photo.jpg", np.clip(sepia, 0, 255).astype(np.uint8))
cv2.imwrite(str(OUT / "old_photo_mask.png"), scratch)               # 划痕位置（实验 8-2 的“标准答案”遮罩）
print("old_photo_mask.png")
