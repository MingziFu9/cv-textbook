# 实验 5-3 训练道路目标检测器：KITTI 400 张训练、100 张验证，4 类（CPU 约 40 分钟，GPU 约 3 分钟）
# 运行前先在 labs/data 下运行一次 python get_kitti_small.py（下载约 390 MB，只需一次）
# %% 第 1 步：看看数据配置文件
from pathlib import Path
from ultralytics import YOLO

DATA = "../data/kitti_small/kitti_small.yaml"
print(Path(DATA).read_text(encoding="utf-8"))

# %% 第 2 步：从 COCO 预训练模型出发，训练 20 轮
model = YOLO("yolo26n.pt")
model.train(data=DATA, epochs=20,                    # ← 改这里：轮次（时间紧可以先用 5）
            imgsz=640, batch=16, device="cpu",       # 有 GPU 就把 device 改成 0
            workers=2, seed=0,
            project=Path("runs").resolve(), name="kitti20", exist_ok=True)

# %% 第 3 步：在验证集上评估，读出 mAP
best = YOLO("runs/kitti20/weights/best.pt")
m = best.val(data=DATA, imgsz=640, verbose=False)
print(f"mAP50 = {m.box.map50:.3f}，mAP50-95 = {m.box.map:.3f}，精确率 = {m.box.mp:.3f}，召回率 = {m.box.mr:.3f}")
for i, name in best.names.items():
    p, r, ap50, ap = m.box.class_result(i)           # 这一类的精确率、召回率、mAP50、mAP50-95
    print(f"  {name:10s} 精确率 {p:.2f}  召回率 {r:.2f}  mAP50 {ap50:.3f}  mAP50-95 {ap:.3f}")

# %% 第 4 步：用训练好的模型检测一张没参加训练的照片
r = best("../data/images/drive_city.jpg", verbose=False)[0]
print("找到：", r.verbose())
r.save("lab5_3_city.jpg")
