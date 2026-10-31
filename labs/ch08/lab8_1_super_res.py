"""实验 8-1 把模糊小图变清楚：最近邻、双三次插值放大与 Real-ESRGAN 超分辨率对比。

在 labs/ch08 目录下运行：python lab8_1_super_res.py（第一次运行会下载约 5 MB 的模型）
"""
# %% 第 1 步：准备一张“标准答案”和它的缩小版
import time

import cv2
import numpy as np
from sr_model import upscale                    # 同文件夹里的超分辨率小工具

big = cv2.imread("../data/images/drive_city.jpg")[160:320, 300:700]   # 一块街景，160×400  ← 改这里
small = cv2.resize(big, (100, 40), interpolation=cv2.INTER_AREA)       # 缩小到 1/4：40×100
print("原图", big.shape, " 小图", small.shape)
cv2.imwrite("lab8_1_small.png", small)

# %% 第 2 步：三种方法放大 4 倍
up_nearest = cv2.resize(small, (400, 160), interpolation=cv2.INTER_NEAREST)  # 最近邻：直接复制像素
up_cubic = cv2.resize(small, (400, 160), interpolation=cv2.INTER_CUBIC)      # 双三次：用周围 16 个像素算
t = time.time()
up_sr = upscale(small)                                                       # Real-ESRGAN 超分辨率
print("超分辨率用时", round(time.time() - t, 2), "秒，结果", up_sr.shape)


# %% 第 3 步：和“标准答案”比一比：平均误差和清晰度
def mean_error(a, b):
    return float(np.abs(a.astype(int) - b.astype(int)).mean())   # 每个数平均差多少


def sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())         # 边缘越多越清晰，数越大


for name, im in [("最近邻", up_nearest), ("双三次", up_cubic), ("超分辨率", up_sr), ("原图", big)]:
    print(f"{name:<6} 平均误差 {mean_error(im, big):5.1f}   清晰度 {sharpness(im):7.1f}")

# %% 第 4 步：拼成一张对比图
row = [cv2.putText(im.copy(), t, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
       for im, t in [(up_nearest, "nearest"), (up_cubic, "bicubic"), (up_sr, "Real-ESRGAN"), (big, "original")]]
cv2.imwrite("lab8_1_compare.jpg", np.vstack(row))
print("对比图已保存：lab8_1_compare.jpg")
