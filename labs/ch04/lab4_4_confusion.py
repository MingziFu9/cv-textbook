# 实验 4-4 看看错在哪里：自己数出混淆矩阵，找出认错的图
# %% 第 1 步：载入实验 4-3 训练好的模型，逐张认验证集
from pathlib import Path
from ultralytics import YOLO

model = YOLO("runs/road5/weights/best.pt")
names = ["car", "cyclist", "pedestrian", "truck", "van"]
matrix = {t: {p: 0 for p in names} for t in names}    # 字典套字典：matrix[真实][预测] = 张数
wrong = []                                            # 认错的图片记在这里
for t in names:
    for f in sorted(Path(f"../data/road_objects/val/{t}").glob("*.jpg")):
        p = model(str(f), imgsz=128, verbose=False)[0]
        pred = p.names[p.probs.top1]                  # 把握最大的那一类
        matrix[t][pred] += 1
        if pred != t:
            wrong.append((f.name, t, pred, float(p.probs.top1conf)))

# %% 第 2 步：打印混淆矩阵（行是真实类别，列是预测类别）
print("真实＼预测".ljust(12) + "".join(n[:6].rjust(8) for n in names) + "   准确率")
for t in names:
    row = matrix[t]
    print(t.ljust(12) + "".join(str(row[p]).rjust(8) for p in names) + f"{row[t] / sum(row.values()):9.1%}")
total = sum(matrix[t][t] for t in names)
print(f"总共认对 {total} 张，认错 {len(wrong)} 张，准确率 {total / (total + len(wrong)):.1%}")

# %% 第 3 步：看看最“自信”的几次认错
wrong.sort(key=lambda w: -w[3])                      # 按把握从大到小排
for name, t, pred, conf in wrong[:8]:
    print(f"{name:16s} 真实 {t:10s} → 认成 {pred:10s}（把握 {conf:.0%}）")
