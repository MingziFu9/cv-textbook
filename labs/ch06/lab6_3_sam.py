"""实验 6-3 点一下分割：用 MobileSAM 点选或框选物体，不用训练就能得到掩膜。

在 labs/ch06 目录下运行：python lab6_3_sam.py
"""
# %% 第 1 步：在黑色轿车上点一个点
import cv2
from ultralytics import SAM, YOLO

sam = SAM("mobile_sam.pt")                       # 手机也能跑的轻量版“分割一切”模型
img = cv2.imread("../data/images/drive_highway.jpg")
r = sam(img, points=[[1180, 500]], labels=[1], verbose=False)[0]   # 点的坐标 (x, y)，1 表示“要这个”  ← 改这里
mask = r.masks.data[0].cpu().numpy()            # True/False 掩膜，和原图一样大
print("掩膜像素数：", int(mask.sum()), " 模型打分：", round(float(r.boxes.conf[0]), 2))
out = r.plot()
cv2.circle(out, (1180, 500), 8, (0, 0, 255), -1)  # 把点过的位置画成红点
cv2.imwrite("lab6_3_point.jpg", out)

# %% 第 2 步：点白色小车，再改用框，比一比
r1 = sam(img, points=[[900, 440]], labels=[1], verbose=False)[0]     # 点在车尾中间
r2 = sam(img, bboxes=[[855, 415, 945, 468]], verbose=False)[0]       # 框住整辆车 (x1, y1, x2, y2)
for name, rr in [("点", r1), ("框", r2)]:
    print(name, "→ 像素数", int(rr.masks.data[0].sum()), " 打分", round(float(rr.boxes.conf[0]), 2))
cv2.imwrite("lab6_3_box.jpg", r2.plot())

# %% 第 3 步：路面、天空、隔离墙都能分，因为 SAM 只管“这是一块”，不管它叫什么
for name, xy in [("路面", [640, 600]), ("天空", [640, 150]), ("隔离墙", [200, 500])]:
    rr = sam(img, points=[xy], labels=[1], verbose=False)[0]
    ratio = rr.masks.data[0].float().mean().item()   # 掩膜占整张图的比例
    print(f"{name}：占画面 {ratio:.1%}")

# %% 第 4 步：先用 YOLO 找车，再把框交给 SAM 描出精确轮廓
det = YOLO("yolo26n.pt")(img, verbose=False)[0]
boxes = det.boxes.xyxy.tolist()                  # 检测框列表
r = sam(img, bboxes=boxes, verbose=False)[0]     # 一次传入多个框，每个框得到一个掩膜
for b, m in zip(boxes, r.masks.data):
    box_area = (b[2] - b[0]) * (b[3] - b[1])     # 框的面积
    print(f"框面积 {box_area:7.0f}  掩膜面积 {int(m.sum()):6d}  车身占框 {m.sum() / box_area:.0%}")
cv2.imwrite("lab6_3_det_sam.jpg", r.plot())
