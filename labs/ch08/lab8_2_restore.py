"""实验 8-2 修补老照片：找出划痕做遮罩，用 OpenCV 修补，再用超分辨率放大。

在 labs/ch08 目录下运行：python lab8_2_restore.py
"""
# %% 第 1 步：自动找划痕——“顶帽”运算找出比周围亮的细小东西
import cv2
import numpy as np
from sr_model import upscale

old = cv2.imread("../data/images/old_photo.jpg")          # 有白色划痕和斑点的“老照片”  ← 改这里
gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)  # 原图 − 抹掉细小亮东西后的图
auto = tophat > 60                                         # 比周围亮 60 以上  ← 改这里
truth = cv2.imread("../data/images/old_photo_mask.png", 0) > 0   # 编者提供的“标准答案”遮罩
inter, union = (auto & truth).sum(), (auto | truth).sum()
print(f"自动找到 {auto.sum()} 个像素，标准答案 {truth.sum()} 个，交并比 {inter / union:.2f}，找全了 {inter / truth.sum():.0%}")
cv2.imwrite("lab8_2_auto_mask.png", auto.astype(np.uint8) * 255)

# %% 第 2 步：用标准答案遮罩修补，比较两种算法
mask = cv2.dilate(truth.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))   # 遮罩向外扩 1 像素，盖住划痕边缘
telea = cv2.inpaint(old, mask, 3, cv2.INPAINT_TELEA)       # 3 是参考周围多大范围
ns = cv2.inpaint(old, mask, 3, cv2.INPAINT_NS)
print("遮罩像素数（扩大后）：", int((mask > 0).sum()))
cv2.imwrite("lab8_2_telea.jpg", telea)
cv2.imwrite("lab8_2_ns.jpg", ns)

# %% 第 3 步：检查修补效果——划痕处还有多亮
for name, im in [("修补前", old), ("TELEA", telea), ("NS", ns)]:
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    print(f"{name:<5} 划痕处平均亮度 {g[truth].mean():5.1f}   全图平均亮度 {g.mean():5.1f}")

# %% 第 4 步：把修补好的行人部分放大 4 倍
crop = telea[160:320, 380:520]                             # 照片中提着袋子的行人  ← 改这里
big_cubic = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
big_sr = upscale(crop)
print("裁剪", crop.shape, "→ 放大后", big_sr.shape)
cv2.imwrite("lab8_2_zoom.jpg", np.hstack([big_cubic, big_sr]))   # 左：插值，右：超分辨率
