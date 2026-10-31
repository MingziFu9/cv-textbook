"""实验 7-3 健身计数器：姿态估计找出关节点，算肘关节角度，用“连续几帧”规则数俯卧撑。

在 labs/ch07 目录下运行：python lab7_3_pushup_counter.py
"""
# %% 第 1 步：姿态估计——一帧图片里找出 17 个人体关键点
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")                  # 名字带 -pose 的是姿态估计模型
video = "../data/videos/pushup_5reps.mp4"        # 连续 5 个俯卧撑  ← 改这里，换成自己录的视频
cap = cv2.VideoCapture(video)
ok, frame = cap.read()
cap.release()
r = model(frame, verbose=False)[0]
kp = r.keypoints.xy[0].tolist()                  # 第一个人的 17 个关键点 (x, y)
names = ["鼻子", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩", "左肘", "右肘",
         "左腕", "右腕", "左髋", "右髋", "左膝", "右膝", "左踝", "右踝"]
for i in [6, 8, 10]:                            # 右肩、右肘、右腕
    print(i, names[i], [round(v) for v in kp[i]])
cv2.imwrite("lab7_3_pose.jpg", r.plot())


# %% 第 2 步：算一个关节的角度（三个点 a-b-c，b 是关节）
def angle(a, b, c):
    ba = np.array(a) - np.array(b)               # 从关节指向 a 的箭头
    bc = np.array(c) - np.array(b)               # 从关节指向 c 的箭头
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))   # 换算成角度


print("右肘角度：", round(angle(kp[6], kp[8], kp[10])), "度（手臂伸直约 180 度）")

# %% 第 3 步：逐帧算角度，用“连续 3 帧”规则数数
UP, DOWN, N = 140, 100, 3                         # 撑起、压下的角度门槛，连续帧数  ← 改这里
state, streak, count, angles = "up", 0, 0, []
for r in model(video, stream=True, verbose=False):
    if r.keypoints is None or len(r.keypoints.xy[0]) == 0:
        angles.append(None)
        continue
    k = r.keypoints.xy[0].tolist()
    a = angle(k[6], k[8], k[10])                  # 右肩-右肘-右腕（右侧对着镜头）
    angles.append(a)
    want = "down" if (state == "up" and a < DOWN) else ("up" if (state == "down" and a > UP) else None)
    streak = streak + 1 if want else 0            # 满足条件的帧连续累计，否则清零
    if streak >= N:                               # 连续 N 帧都满足，才算真的换了状态
        state, streak = want, 0
        if state == "up":                         # 从“压下”回到“撑起”，算完成一个
            count += 1
            print(f"第 {len(angles)} 帧完成第 {count} 个")
print("一共", count, "个俯卧撑")

# %% 第 4 步：画出角度曲线，看清楚“动作是一段时间”
h, w = 200, len(angles) * 2
chart = np.full((h, w, 3), 255, np.uint8)
for y_deg, color in [(UP, (0, 160, 0)), (DOWN, (0, 0, 220))]:   # 两条门槛线
    cv2.line(chart, (0, h - int(y_deg)), (w, h - int(y_deg)), color, 1)
pts = [(i * 2, h - int(a)) for i, a in enumerate(angles) if a is not None]
cv2.polylines(chart, [np.array(pts)], False, (200, 0, 0), 2)   # 角度随时间的曲线
cv2.imwrite("lab7_3_angles.png", chart)
valid = [a for a in angles if a is not None]
print("角度最小", round(min(valid)), "最大", round(max(valid)), "，曲线已保存")
