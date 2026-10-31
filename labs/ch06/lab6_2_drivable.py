"""实验 6-2 分出可行驶路面：用道路语义分割模型给行车照片的每个像素分类，算路面占比。

在 labs/ch06 目录下运行：python lab6_2_drivable.py
"""
# %% 第 1 步：语义分割，每个像素一个类别编号
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo26n-sem.pt")                  # 在城市道路数据上训练的语义分割模型，认识 19 类
print("模型认识的类别：", model.names)

img = cv2.imread("../data/images/drive_city.jpg")   # ← 改这里，换成 drive_highway.jpg 试试
r = model(img, verbose=False)[0]
sem = r.semantic_mask.data.cpu().numpy()        # 和原图一样大的二维数组，每个数是类别编号
print("结果形状：", sem.shape, "  左下角像素的类别：", model.names[int(sem[-1, 0])])
cv2.imwrite("lab6_2_sem.jpg", r.plot())         # 不同类别涂上不同颜色

# %% 第 2 步：统计每一类占多大面积
ids, counts = np.unique(sem, return_counts=True)   # 出现过的编号，以及各有多少个像素
for i, n in sorted(zip(ids, counts), key=lambda t: -t[1]):  # 按像素数从多到少排
    print(f"{model.names[int(i)]:<14}{n / sem.size * 100:5.1f}%")

# %% 第 3 步：把路面单独涂成绿色
road = (sem == 0)                               # 编号 0 是 road（路面），得到 True/False 掩膜
show = img.copy()
show[road] = (0.5 * show[road] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)  # 半透明绿色
print("整张图中路面占", round(road.mean() * 100, 1), "%")
cv2.imwrite("lab6_2_road.jpg", show)

# %% 第 4 步：只看车头正前方的区域，判断能不能往前开
for name in ["drive_city", "street_crossing", "drive_highway"]:    # 三张行车照片轮流判断
    sem = model(f"../data/images/{name}.jpg", verbose=False)[0].semantic_mask.data.cpu().numpy()
    h, w = sem.shape
    front = sem[int(h * 0.6):, int(w * 0.35):int(w * 0.65)]   # 画面下部中间 = 车头正前方  ← 改这里
    road_ratio = (front == 0).mean()                          # 路面占比
    blocked = np.isin(front, [11, 12, 13, 14, 15, 17, 18]).mean()   # 人、骑车人、各种车占比
    if blocked > 0.05:                          # 阈值  ← 改这里
        tip = "正前方有人或车，注意减速"
    elif road_ratio < 0.7:
        tip = "正前方不全是路面，请注意"
    else:
        tip = "前方路面空旷"
    print(f"{name:<16}路面 {road_ratio:4.0%}  人或车 {blocked:4.0%}  → {tip}")
