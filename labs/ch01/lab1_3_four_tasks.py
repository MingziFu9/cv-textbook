# 实验 1-3 一行代码体验四种视觉任务
# %% 第 1 步：准备照片（第一次运行会自动下载 4 个模型，每个约 5～7 MB）
from ultralytics import YOLO

PHOTO = "../data/images/street_crossing.jpg"     # ← 改这里：换成 drive_city.jpg 试试

# %% 第 2 步：分类——整张照片是什么？
r = YOLO("yolo26n-cls.pt")(PHOTO, verbose=False)[0]
for i in r.probs.top5:                            # 把握最大的 5 个类别
    print(f"分类：{r.names[i]:20s} 把握 {float(r.probs.data[i]):.1%}")

# %% 第 3 步：检测——有什么？在哪里？
r = YOLO("yolo26n.pt")(PHOTO, verbose=False)[0]
print("检测：找到", len(r.boxes), "个框 →", r.verbose())
r.save("lab1_3_detect.jpg")

# %% 第 4 步：分割——每个东西的精确轮廓
r = YOLO("yolo26n-seg.pt")(PHOTO, verbose=False)[0]
print("分割：找到", len(r.masks), "个轮廓，每个轮廓是一张", tuple(r.masks.data.shape[1:]), "的黑白图")
r.save("lab1_3_segment.jpg")

# %% 第 5 步：姿态估计——每个人的关节在哪里
r = YOLO("yolo26n-pose.pt")(PHOTO, verbose=False)[0]
print("姿态：找到", len(r.keypoints), "个人，每人", r.keypoints.xy.shape[1], "个关键点")
r.save("lab1_3_pose.jpg")
