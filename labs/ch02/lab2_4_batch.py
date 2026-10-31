# 实验 2-4 批量处理：把一个文件夹里的照片统一改尺寸、加水印、另存
# %% 第 1 步：列出文件夹里所有 .jpg 图片
from pathlib import Path
import cv2

src = Path("../data/images")                 # ← 改这里：换成你自己的照片文件夹
dst = Path("lab2_4_output")
dst.mkdir(exist_ok=True)                     # 建一个输出文件夹，已有就不管
files = sorted(src.glob("*.jpg"))            # 找出所有 .jpg 文件，按名字排好
print("找到", len(files), "张图片：", [f.name for f in files])

# %% 第 2 步：写一个函数，处理一张图片
def process(path, width=640, text="CV Lab"):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    new_h = int(h * width / w)                         # 宽改成 width，高按比例算
    small = cv2.resize(img, (width, new_h))
    cv2.putText(small, text, (10, new_h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)               # 左下角写白色水印
    return small

# %% 第 3 步：用 for 循环处理每一张
for f in files:
    out = process(f, width=640, text="CV Lab 2026")    # ← 改这里：宽度和水印文字
    cv2.imwrite(str(dst / f.name), out)
    print(f"{f.name:20s} → {out.shape[1]}×{out.shape[0]}")
print("全部完成，结果在", dst)
