# 实验 2-3 用颜色找车道线：只看下半部分，用 HSV 阈值找出白线和黄线
# %% 第 1 步：读图，只留下画面下半部分（车道只在路面上）
import cv2

img = cv2.imread("../data/images/lane_yellow.jpg")   # ← 改这里：换成 lane_white.jpg
h, w = img.shape[:2]
roi = img[h // 2 + 50:, :]          # 从中线再往下 50 行开始，到最底下，所有列
print("感兴趣区域的形状：", roi.shape)

# %% 第 2 步：换到 HSV，分别找黄色和白色
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
yellow = cv2.inRange(hsv, (15, 80, 120), (35, 255, 255))   # 黄：色调 15～35，比较鲜艳  ← 改这里
white = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))      # 白：不管色调，不鲜艳，很亮  ← 改这里
print(f"黄色像素 {cv2.countNonZero(yellow)} 个，白色像素 {cv2.countNonZero(white)} 个")

# %% 第 3 步：把两张黑白图合起来，再把车道线涂成红色
lanes = cv2.bitwise_or(yellow, white)     # 黄或白，都算车道线
show = img.copy()
show[h // 2 + 50:][lanes == 255] = (0, 0, 255)   # 在原图对应位置涂红
cv2.imwrite("lab2_3_mask.png", lanes)
cv2.imwrite("lab2_3_lanes.jpg", show)

# %% 第 4 步：试试夜间画面：同样的门槛还管用吗？
dark = cv2.convertScaleAbs(img, alpha=0.35, beta=0)      # 把画面压暗，模拟夜里
hsv2 = cv2.cvtColor(dark[h // 2 + 50:, :], cv2.COLOR_BGR2HSV)
n_y = cv2.countNonZero(cv2.inRange(hsv2, (15, 80, 120), (35, 255, 255)))
n_w = cv2.countNonZero(cv2.inRange(hsv2, (0, 0, 200), (180, 40, 255)))
print(f"夜间：黄色像素 {n_y} 个，白色像素 {n_w} 个")
