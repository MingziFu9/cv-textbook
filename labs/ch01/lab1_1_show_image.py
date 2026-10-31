# 实验 1-1 打开 Notebook，显示第一张图片
# %% 第 1 步：请出两个工具
import cv2                          # OpenCV：读图、处理图片的工具箱
import matplotlib.pyplot as plt     # Matplotlib：在 Notebook 里画图、显示图片

# %% 第 2 步：读入一张行车照片
img = cv2.imread("../data/images/drive_city.jpg")   # ← 改这里：换成别的图片
print("图片的形状：", img.shape)                     # (高, 宽, 通道数)

# %% 第 3 步：显示出来
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV 按“蓝绿红”存，显示前要换成“红绿蓝”
plt.figure(figsize=(10, 3))                 # 画布大小：宽 10、高 3（英寸）
plt.imshow(rgb)                             # 把图片放到画布上
plt.axis("off")                             # 不显示坐标轴
plt.title("drive_city.jpg")                 # 标题
plt.savefig("lab1_1_result.png", dpi=80, bbox_inches="tight")   # 存一份，方便交报告
plt.show()
