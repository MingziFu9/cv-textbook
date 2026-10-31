# 实验 5-4 标注自己的数据：检查导出的 YOLO 标注是否正确，并把框画出来
# %% 第 1 步：设定文件夹（换成你自己导出的 images 和 labels 文件夹）
from pathlib import Path
import cv2

IMAGES = Path("../data/kitti_sample")          # ← 改这里：你的图片文件夹
LABELS = Path("../data/kitti_sample")          # ← 改这里：你的标注文件夹（.txt 与图片同名）
NAMES = {0: "car", 1: "van", 2: "truck", 3: "pedestrian", 5: "cyclist"}   # ← 改这里：你的类别（这里是 KITTI 原始编号）
out = Path("lab5_4_check"); out.mkdir(exist_ok=True)

# %% 第 2 步：逐张检查：每行是否 5 个数、坐标是否在 0～1 之间
problems = 0
for img_path in sorted(IMAGES.glob("*.jpg")):
    txt = LABELS / (img_path.stem + ".txt")
    if not txt.exists():
        print("缺少标注：", img_path.name); problems += 1; continue
    img = cv2.imread(str(img_path)); H, W = img.shape[:2]
    for n, line in enumerate(txt.read_text().splitlines(), 1):
        v = line.split()
        if len(v) != 5 or not all(0 <= float(x) <= 1 for x in v[1:]):
            print(f"{txt.name} 第 {n} 行格式不对：{line}"); problems += 1; continue
        c, x, y, w, h = int(v[0]), *map(float, v[1:])
        p1, p2 = (int((x - w / 2) * W), int((y - h / 2) * H)), (int((x + w / 2) * W), int((y + h / 2) * H))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        cv2.putText(img, NAMES.get(c, str(c)), (p1[0], p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(out / img_path.name), img)
print("检查完成，发现问题", problems, "处；画好框的图在", out)
