# 实验 4-2 整理道路目标数据集：从行车照片按标注框裁出小图，统计各类数量
# %% 第 1 步：读一张 KITTI 照片和它的标注文件
import cv2
from pathlib import Path

NAMES = {0: "car", 1: "van", 2: "truck", 3: "pedestrian",
         4: "person_sitting", 5: "cyclist", 6: "tram", 7: "misc"}   # 字典：编号 → 类别名
img = cv2.imread("../data/kitti_sample/000380.jpg")
H, W = img.shape[:2]
lines = Path("../data/kitti_sample/000380.txt").read_text().splitlines()
print(f"照片 {W}×{H}，标注了 {len(lines)} 个物体；第一行：{lines[0]}")

# %% 第 2 步：把比例坐标换成像素坐标，裁出每个物体
out = Path("lab4_2_crops")
out.mkdir(exist_ok=True)
for k, line in enumerate(lines):
    c, x, y, w, h = line.split()                          # 类别、中心 x、中心 y、宽、高（都是 0～1 的比例）
    x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
    x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
    name = NAMES[int(c)]
    cv2.imwrite(str(out / f"{k}_{name}.jpg"), img[y1:y2, x1:x2])
    print(f"{k}: {name:11s} 左上 ({x1}, {y1})  右下 ({x2}, {y2})  大小 {x2 - x1}×{y2 - y1}")

# %% 第 3 步：统计整理好的数据集里每类有多少张
root = Path("../data/road_objects")                       # 编者用同样方法裁好的 5 类小图
for split in ["train", "val"]:
    counts = {}                                           # 空字典
    for folder in sorted((root / split).iterdir()):       # 每个类别一个文件夹
        counts[folder.name] = len(list(folder.glob("*.jpg")))
    print(split, counts, "合计", sum(counts.values()))

# %% 第 4 步：看看每类小图的平均宽高
for folder in sorted((root / "train").iterdir()):
    sizes = [cv2.imread(str(f)).shape[:2] for f in folder.glob("*.jpg")]
    avg_h = sum(s[0] for s in sizes) / len(sizes)
    avg_w = sum(s[1] for s in sizes) / len(sizes)
    print(f"{folder.name:11s} 平均 宽 {avg_w:5.1f} × 高 {avg_h:5.1f}")
