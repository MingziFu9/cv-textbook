"""实验 6-1 抠图换背景：用实例分割找出人，换成蓝底、白底和新的街景背景。

在 labs/ch06 目录下运行：python lab6_1_cutout.py
"""
# %% 第 1 步：用实例分割模型找出图中的物体
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import ASSETS          # Ultralytics 自带的示例图片所在文件夹

img_path = ASSETS / "bus.jpg"                  # 街头公交车前的几个人  ← 改这里，换成自己的照片
img = cv2.imread(str(img_path))
h, w = img.shape[:2]
print("图片大小：", w, "×", h)

model = YOLO("yolo26n-seg.pt")                 # 名字带 -seg 的是实例分割模型
r = model(img, retina_masks=True, verbose=False)[0]   # retina_masks=True：掩膜和原图一样大
for i, (c, conf) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())):
    print(i, r.names[int(c)], round(conf, 2), "掩膜像素数：", int(r.masks.data[i].sum()))
cv2.imwrite("lab6_1_seg.jpg", r.plot())        # 框 + 彩色掩膜一起画出来

# %% 第 2 步：取出第一个人的掩膜，看看它长什么样
person_ids = [i for i, c in enumerate(r.boxes.cls.tolist()) if r.names[int(c)] == "person"]
k = person_ids[0]                              # 第几个人  ← 改这里
mask = r.masks.data[k].cpu().numpy()           # 0 和 1 组成的二维数组，和原图一样大
print("掩膜形状：", mask.shape, "取值：", np.unique(mask))
print("这个人占整张图的", round(mask.mean() * 100, 1), "%")
cv2.imwrite("lab6_1_mask.png", (mask * 255).astype(np.uint8))   # 1 → 白色，0 → 黑色

# %% 第 3 步：换成蓝底证件照
mask3 = mask[:, :, None]                        # 变成 h×w×1，才能和 3 个通道的图片相乘
blue = np.zeros_like(img)
blue[:] = (219, 142, 67)                       # 证件照蓝（BGR）  ← 改这里，例如白底 (255,255,255)
out = img * mask3 + blue * (1 - mask3)          # 人身上用原图，其他地方用蓝色
out = out.astype(np.uint8)
x1, y1, x2, y2 = r.boxes.xyxy[k].int().tolist() # 用检测框把这个人裁出来
cv2.imwrite("lab6_1_blue.jpg", out[y1:y2, x1:x2])

# %% 第 4 步：把人“搬”到另一张街景照片上
bg = cv2.imread("../data/images/drive_city.jpg")   # 新背景  ← 改这里
bg = cv2.resize(bg, (w, h))                     # 背景缩放成和原图一样大
all_people = np.zeros((h, w), np.float32)
for i in person_ids:                            # 把所有人的掩膜合在一起
    all_people = np.maximum(all_people, r.masks.data[i].cpu().numpy())
soft = cv2.GaussianBlur(all_people, (7, 7), 0)[:, :, None]   # 边缘稍微羽化，看起来更自然
out = (img * soft + bg * (1 - soft)).astype(np.uint8)
cv2.imwrite("lab6_1_newbg.jpg", out)
print("一共搬了", len(person_ids), "个人，结果已保存")
