# 实验 4-1 认出 1000 种东西：用预训练分类模型识别照片，读 Top-5
# %% 第 1 步：载入预训练分类模型（第一次运行会自动下载，约 6 MB）
import cv2
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")
print("模型认识", len(model.names), "种东西，前 5 种是：", [model.names[i] for i in range(5)])

# %% 第 2 步：从街景照片里裁出三样东西
img = cv2.imread("../data/images/drive_city.jpg")
crops = {                                   # 字典：名字 → 裁出的小图
    "白色厢式车": img[150:240, 385:530],     # ← 改这里：换一块区域试试
    "停着的轿车": img[190:370, 0:300],
    "推车的老人": img[180:325, 885:955],
}

# %% 第 3 步：逐个识别，打印 Top-5
for name, crop in crops.items():
    r = model(crop, verbose=False)[0]          # 直接把图片（数字表）交给模型
    print(f"【{name}】")
    for i in r.probs.top5:                     # 把握最大的 5 个类别的编号
        print(f"    {r.names[i]:22s} {float(r.probs.data[i]):6.1%}")
