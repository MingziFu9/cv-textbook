# 实验 5-2 调门槛做对比：不同置信度门槛下，数一数认对、误检、漏检
# %% 第 1 步：读标准答案（KITTI 人工标注），只看“车”和“人”两大类
from pathlib import Path
from ultralytics import YOLO

W, H = 1242, 375
KITTI = {0: "car", 1: "car", 3: "person", 5: "person"}     # 轿车、厢式车算车；行人、骑车人算人
truth = []                                                 # 标准答案：[(类别, x1, y1, x2, y2), ...]
for line in Path("../data/kitti_sample/000380.txt").read_text().splitlines():
    c, x, y, w, h = map(float, line.split())
    if int(c) in KITTI:
        truth.append((KITTI[int(c)], (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
print("标准答案：", len(truth), "个物体")

# %% 第 2 步：写一个算交并比（IoU）的函数
def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))          # 重叠部分的宽
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))          # 重叠部分的高
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union

# %% 第 3 步：门槛压到很低，先拿到所有候选框
r = YOLO("yolo26n.pt")("../data/kitti_sample/000380.jpg", conf=0.01, verbose=False)[0]
preds = [(r.names[int(c)], float(s), *b) for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy.tolist())]
preds = [p for p in preds if p[0] in ("car", "bus", "truck", "person")]
preds = [("person" if p[0] == "person" else "car", *p[1:]) for p in preds]   # 公交车、卡车也算车
print("候选框：", len(preds), "个")

# %% 第 4 步：换几个门槛，数认对（TP）、误检（FP）、漏检（FN）
for th in [0.1, 0.25, 0.5, 0.7]:                           # ← 改这里：试试别的门槛
    kept = [p for p in preds if p[1] >= th]
    used, tp = set(), 0                                    # used：已经被认领的标准答案
    for name, score, *box in sorted(kept, key=lambda p: -p[1]):   # 置信度高的先认领
        ious = [iou(box, t[1:]) if t[0] == name and i not in used else 0 for i, t in enumerate(truth)]
        i = ious.index(max(ious))                          # 和哪个标准答案重叠最多
        if ious[i] >= 0.5:                                 # 交并比达到 0.5 才算认对
            used.add(i); tp += 1
    fp, fn = len(kept) - tp, len(truth) - tp
    print(f"门槛 {th:.2f}：框 {len(kept):2d} 个，认对 {tp}，误检 {fp}，漏检 {fn}，"
          f"精确率 {tp / max(1, len(kept)):.0%}，召回率 {tp / len(truth):.0%}")
