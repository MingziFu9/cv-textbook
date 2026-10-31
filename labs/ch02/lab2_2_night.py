# 实验 2-2 让夜间行车画面看得清：调亮度、直方图均衡化、去噪
# %% 第 1 步：读入夜间画面，算平均亮度
import cv2
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei"]  # 图上显示中文

img = cv2.imread("../data/images/drive_night.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"平均亮度：{gray.mean():.1f}（0 最暗，255 最亮）")
print(f"亮度低于 50 的像素占 {(gray < 50).mean():.1%}")

# %% 第 2 步：直接调亮度和对比度：新值 = 原值 × alpha + beta
bright = cv2.convertScaleAbs(img, alpha=2.0, beta=20)   # ← 改这里：alpha 管对比度，beta 管亮度
g2 = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)
print(f"调亮后平均亮度：{g2.mean():.1f}，变成 255 的像素占 {(g2 == 255).mean():.1%}")
cv2.imwrite("lab2_2_bright.jpg", bright)

# %% 第 3 步：直方图均衡化（只拉亮度 V，不动颜色）
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])     # 第 2 号通道是亮度 V
equal = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
g3 = cv2.cvtColor(equal, cv2.COLOR_BGR2GRAY)
print(f"均衡化后平均亮度：{g3.mean():.1f}")
cv2.imwrite("lab2_2_equal.jpg", equal)

# %% 第 4 步：均衡化把噪点也放大了，用中值滤波去噪
clean = cv2.medianBlur(equal, 3)           # ← 改这里：窗口大小 3、5、7（必须是奇数）
g4 = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
before = g3[300:360, 450:600].std()       # 取路面上的一块（本来应该很平整），算“毛糙程度”（标准差）
after = g4[300:360, 450:600].std()
print(f"路面这块的毛糙程度：去噪前 {before:.1f}，去噪后 {after:.1f}")
cv2.imwrite("lab2_2_clean.jpg", clean)

# %% 第 5 步：画出调整前后的直方图
plt.figure(figsize=(10, 3))
plt.hist(gray.ravel(), bins=256, range=(0, 256), color="gray", alpha=0.6, label="原图")
plt.hist(g3.ravel(), bins=256, range=(0, 256), color="orange", alpha=0.6, label="均衡化后")
plt.xlabel("亮度"); plt.ylabel("像素个数"); plt.legend()
plt.savefig("lab2_2_hist.png", dpi=100, bbox_inches="tight")
plt.show()
