# 实验 1-2 图片变数字
# %% 第 1 步：读图，看看它有多大
import cv2

img = cv2.imread("../data/images/drive_city.jpg")
h, w, c = img.shape                  # 高、宽、通道数
print(f"高 {h} 像素，宽 {w} 像素，{c} 个通道，一共 {h * w * c} 个数字")
print("数字的类型：", img.dtype)     # uint8：0～255 的整数

# %% 第 2 步：取一个像素的值
y, x = 100, 600                      # ← 改这里：第 100 行、第 600 列
b, g, r = img[y, x]                  # OpenCV 的顺序是 蓝、绿、红
print(f"第 {y} 行第 {x} 列：红 {r}，绿 {g}，蓝 {b}")
print("左上角 3×3 个像素的蓝色通道：")
print(img[0:3, 0:3, 0])

# %% 第 3 步：把一块区域涂成红色，再存成新图片
painted = img.copy()                 # 复制一份，不改原图
painted[250:330, 560:700] = (0, 0, 255)     # 行 250～329、列 560～699 改成纯红（蓝0 绿0 红255）  ← 改这里
cv2.imwrite("lab1_2_painted.jpg", painted)
print("涂好的图片存为 lab1_2_painted.jpg")

# %% 第 4 步：变成灰度图，看看通道数
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("灰度图的形状：", gray.shape, " 同一位置的灰度值：", gray[y, x])
