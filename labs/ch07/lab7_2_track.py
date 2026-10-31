"""实验 7-2 行车视频跟踪与提醒：给每辆车发号码牌，看它是越来越近还是越来越远；
再在行人视频中划一块区域，有人进入就标红。

在 labs/ch07 目录下运行：python lab7_2_track.py
"""
# %% 第 1 步：跟踪——每辆车一个号码牌（ID）
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
video = "../data/videos/highway_12s.mp4"         # ← 改这里
history = {}                                      # 号码牌 → 每一帧框的面积
for i, r in enumerate(model.track(video, stream=True, persist=True, classes=[2, 5, 7], verbose=False)):
    if r.boxes.id is None:                        # 这一帧没有跟踪到任何车
        continue
    for tid, (x1, y1, x2, y2) in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.tolist()):
        history.setdefault(tid, []).append((i, (x2 - x1) * (y2 - y1)))
for tid, h in history.items():
    print(f"车 {tid}：从第 {h[0][0]} 帧出现到第 {h[-1][0]} 帧，共 {len(h)} 帧")

# %% 第 2 步：比较 1 秒前后框的面积，判断越来越近还是越来越远
fps = 25
for tid, h in history.items():
    for k in range(fps, len(h), fps):             # 每隔 1 秒看一次
        (f0, a0), (f1, a1) = h[k - fps], h[k]     # 1 秒前和现在
        ratio = a1 / a0                           # 面积变化倍数
        trend = "变近" if ratio > 1.2 else ("变远" if ratio < 0.8 else "差不多")  # 阈值  ← 改这里
        print(f"车 {tid} 第 {f1:3d} 帧：面积 {a0:6.0f} → {a1:6.0f}，×{ratio:.2f} {trend}")

# %% 第 3 步：把提醒画进视频——框的高度超过画面的 1/4 就标黄“很近”
cap = cv2.VideoCapture(video)
w, h = int(cap.get(3)), int(cap.get(4))
cap.release()
writer = cv2.VideoWriter("lab7_2_highway.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
near_frames = 0
model = YOLO("yolo26n.pt")                        # 重新加载，号码牌重新从 1 开始，和第 1 步一致
for r in model.track(video, stream=True, persist=True, classes=[2, 5, 7], verbose=False):
    frame = r.orig_img.copy()                     # 这一帧原图
    ids = r.boxes.id.int().tolist() if r.boxes.id is not None else []
    near = False
    for tid, (x1, y1, x2, y2) in zip(ids, r.boxes.xyxy.int().tolist()):
        close = (y2 - y1) > h / 4                  # 框高超过画面 1/4  ← 改这里
        near = near or close
        color = (0, 255, 255) if close else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if near:
        near_frames += 1
        cv2.putText(frame, "CAR CLOSE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    writer.write(frame)
writer.release()
print("有车很近的帧数：", near_frames, "，约", round(near_frames / fps, 1), "秒")

# %% 第 4 步：区域闯入——在行人视频中划一块区域，有人进入就标红
model2 = YOLO("yolo26n.pt")                       # 换一段视频，重新加载模型，号码牌从 1 开始
video2 = "../data/videos/people_2s.mp4"
zone = np.array([[200, 150], [440, 150], [440, 330], [200, 330]])   # 区域的四个角  ← 改这里
entered = set()                                   # 进入过区域的号码牌（集合：自动去重）
writer = cv2.VideoWriter("lab7_2_zone.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
for r in model2.track(video2, stream=True, persist=True, classes=[0], verbose=False):
    frame = r.orig_img.copy()
    cv2.polylines(frame, [zone], True, (255, 0, 0), 2)
    ids = r.boxes.id.int().tolist() if r.boxes.id is not None else []
    inside_now = 0
    for tid, (x1, y1, x2, y2) in zip(ids, r.boxes.xyxy.int().tolist()):
        foot = ((x1 + x2) // 2, y2)               # 用“脚下”那个点判断位置
        inside = cv2.pointPolygonTest(zone, foot, False) >= 0   # 点在多边形里返回 ≥ 0
        if inside:
            entered.add(tid)
            inside_now += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255) if inside else (0, 200, 0), 2)
    cv2.putText(frame, f"in zone: {inside_now}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    writer.write(frame)
writer.release()
print("进入过区域的人数：", len(entered), " 号码牌：", sorted(entered))
