# 实验 5-1 找出行车画面里的车辆和行人：检测、画框、按类别数数
# %% 第 1 步：载入预训练检测模型，看一张行车照片
from ultralytics import YOLO

model = YOLO("yolo26n.pt")                         # 在 COCO 数据集上训练过，认识 80 类
print("模型认识", len(model.names), "类，其中有：", [model.names[i] for i in [0, 1, 2, 3, 5, 7, 9, 11]])
r = model("../data/images/drive_city.jpg", verbose=False)[0]   # ← 改这里：换一张照片
print("找到", len(r.boxes), "个框")

# %% 第 2 步：逐个看每个框：类别、置信度、位置
for box in r.boxes:
    name = r.names[int(box.cls)]                   # 类别编号 → 类别名
    conf = float(box.conf)                         # 置信度
    x1, y1, x2, y2 = box.xyxy[0].int().tolist()    # 左上角、右下角
    print(f"{name:10s} {conf:.2f}  左上 ({x1}, {y1})  右下 ({x2}, {y2})  大小 {x2 - x1}×{y2 - y1}")

# %% 第 3 步：按类别数一数，并保存画好框的图
counts = {}
for c in r.boxes.cls.int().tolist():
    name = r.names[c]
    counts[name] = counts.get(name, 0) + 1         # get：没有这个键时当作 0
print("统计：", counts)
r.save("lab5_1_city.jpg")
