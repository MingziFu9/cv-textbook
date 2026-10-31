# 实验 4-3 训练车辆行人类型分类器：在预训练模型上训练 10 轮（CPU 约 1～3 分钟）
# %% 第 1 步：载入预训练模型，开始训练
from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolo26n-cls.pt")                    # 从“会认 1000 种东西”的模型出发
model.train(
    data="../data/road_objects",                  # 数据集文件夹：里面有 train、val，每类一个子文件夹
    epochs=10,                                    # ← 改这里：训练几轮
    imgsz=128,                                    # ← 改这里：图片缩放到多大
    batch=32,                                     # 每次看 32 张
    device="cpu", workers=2, seed=0,
    project=Path("runs").resolve(), name="road5", exist_ok=True,  # 结果存到本文件夹下的 runs/road5
)

# %% 第 2 步：在验证集上评估
best = YOLO("runs/road5/weights/best.pt")         # 训练中验证准确率最高的那一版
metrics = best.val(data="../data/road_objects", imgsz=128, verbose=False)
print(f"验证集 Top-1 准确率：{metrics.top1:.1%}，Top-5 准确率：{metrics.top5:.1%}")

# %% 第 3 步：用训练好的模型认一认街景里的白色厢式车
import cv2
crop = cv2.imread("../data/images/drive_city.jpg")[150:240, 385:530]
r = best(crop, verbose=False)[0]
for i in r.probs.top5:
    print(f"    {r.names[i]:11s} {float(r.probs.data[i]):6.1%}")
