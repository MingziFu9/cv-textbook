# 实验 2-1 证件照小工具：裁剪成一寸比例、缩放、把蓝底换成白底
# %% 第 1 步：读入照片，看看大小
import cv2

img = cv2.imread("../data/images/portrait_blue.jpg")   # ← 改这里：换成自己的蓝底照片
h, w = img.shape[:2]                                   # 只要高和宽
print(f"原图：宽 {w}，高 {h}，宽高比 {w / h:.3f}")

# %% 第 2 步：裁剪成一寸照的比例（宽:高 = 25:35）
new_w = int(h * 25 / 35)             # 高不变，按比例算出新的宽
x0 = (w - new_w) // 2                # 左右各切掉一样多，人像保持居中
crop = img[:, x0:x0 + new_w]         # 取表格的一块：所有行，第 x0 列到第 x0+new_w 列
print("裁剪后：", crop.shape)

# %% 第 3 步：缩放到一寸照的标准像素（295×413）
photo = cv2.resize(crop, (295, 413))         # 注意先写宽、再写高
print("缩放后：", photo.shape)
cv2.imwrite("lab2_1_blue.jpg", photo)

# %% 第 4 步：用 HSV 找出蓝色背景
hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)       # 换到 HSV 颜色空间
lower = (100, 100, 50)                             # 蓝色的下限：色调 H、饱和度 S、亮度 V  ← 改这里
upper = (130, 255, 255)                            # 蓝色的上限
mask = cv2.inRange(hsv, lower, upper)              # 在范围内的像素记为 255（白），否则 0（黑）
print(f"蓝色像素有 {cv2.countNonZero(mask)} 个，占 {cv2.countNonZero(mask) / mask.size:.1%}")
cv2.imwrite("lab2_1_mask.png", mask)

# %% 第 5 步：把蓝色的地方换成白色，存图
white = photo.copy()
white[mask == 255] = (255, 255, 255)       # 掩膜是白的地方，颜色改成白色（蓝, 绿, 红）  ← 改这里
cv2.imwrite("lab2_1_white.jpg", white)
print("白底证件照存为 lab2_1_white.jpg")
