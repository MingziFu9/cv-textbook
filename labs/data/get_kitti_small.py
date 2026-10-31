"""准备第 4、5 章用的 KITTI 道路数据（小样本）。

在 labs/data 目录下运行：python get_kitti_small.py
第一次运行会从 Ultralytics 的 GitHub 下载 KITTI 目标检测数据集（约 390 MB，已转换成 YOLO 格式），
然后挑出一小部分，生成两个文件夹：
  kitti_small/          第 5 章检测用：训练 400 张、验证 100 张，4 类（car、truck、pedestrian、cyclist）
  road_objects/         第 4 章分类用：从标注框裁出的小图，5 类，每类训练 160 张、验证 40 张
KITTI 数据集许可：CC BY-NC-SA 3.0（仅限非商业用途），引用见 labs/data/README.md。
"""
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip"
ZIP = HERE / "kitti.zip"
RAW = HERE / "kitti_raw"
random.seed(42)

# KITTI 原来 8 类：0 car 1 van 2 truck 3 pedestrian 4 person_sitting 5 cyclist 6 tram 7 misc
DET_MAP = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 3}          # 检测：轿车和厢式车合成 car
DET_NAMES = ["car", "truck", "pedestrian", "cyclist"]
CLS_NAMES = {0: "car", 1: "van", 2: "truck", 3: "pedestrian", 5: "cyclist"}   # 分类：5 类


def download():
    if RAW.exists():
        return
    if not ZIP.exists():
        print("正在下载 KITTI（约 390 MB），请耐心等待……")
        urllib.request.urlretrieve(URL, ZIP)
    print("正在解压……")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(RAW)


def read_labels(txt):
    rows = []
    for line in txt.read_text().splitlines():
        if line.strip():
            c, x, y, w, h = line.split()
            rows.append((int(c), float(x), float(y), float(w), float(h)))
    return rows


def make_detection():
    out = HERE / "kitti_small"
    if out.exists():
        shutil.rmtree(out)
    labels = sorted((RAW / "labels/train").glob("*.txt"))
    useful = [t for t in labels if any(c in DET_MAP for c, *_ in read_labels(t))]
    random.shuffle(useful)
    for split, part in (("train", useful[:400]), ("val", useful[400:500])):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for t in part:
            img = Image.open(RAW / "images/train" / (t.stem + ".png")).convert("RGB")
            img.save(out / "images" / split / (t.stem + ".jpg"), quality=90)   # PNG 转 JPG，体积小很多
            rows = [f"{DET_MAP[c]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in read_labels(t) if c in DET_MAP]
            (out / "labels" / split / t.name).write_text("\n".join(rows) + "\n")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(DET_NAMES))
    (out / "kitti_small.yaml").write_text(
        f"# KITTI 小样本（CC BY-NC-SA 3.0），由 get_kitti_small.py 生成\npath: {out.resolve()}\n"
        f"train: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    print("检测数据：", out, "训练 400 张，验证 100 张")


def make_classification(per_class=200, min_size=32):
    out = HERE / "road_objects"
    if out.exists():
        shutil.rmtree(out)
    crops = {n: [] for n in CLS_NAMES.values()}
    labels = sorted((RAW / "labels/train").glob("*.txt"))
    random.shuffle(labels)
    for t in labels:
        for c, x, y, w, h in read_labels(t):
            name = CLS_NAMES.get(c)
            if name and len(crops[name]) < per_class:
                crops[name].append((t.stem, x, y, w, h))
        if all(len(v) >= per_class for v in crops.values()):
            break
    for name, items in crops.items():
        for k, (stem, x, y, w, h) in enumerate(items):
            img = Image.open(RAW / "images/train" / (stem + ".png")).convert("RGB")
            W, H = img.size
            box = (int((x - w / 2) * W), int((y - h / 2) * H), int((x + w / 2) * W), int((y + h / 2) * H))
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                continue
            crop = img.crop(box)
            scale = min_size / min(crop.size)
            if scale > 1:                                    # 太小的放大到短边 32 像素
                crop = crop.resize((round(crop.width * scale), round(crop.height * scale)))
            split = "train" if k < per_class * 0.8 else "val"
            d = out / split / name
            d.mkdir(parents=True, exist_ok=True)
            crop.save(d / f"{stem}_{k}.jpg", quality=90)
    for split in ("train", "val"):
        print("分类数据", split, {p.name: len(list(p.glob("*.jpg"))) for p in sorted((out / split).iterdir())})


if __name__ == "__main__":
    download()
    make_detection()
    make_classification()
    print("完成。不再需要原始数据的话，可以删除 kitti.zip 和 kitti_raw 文件夹。")
