"""编者用来生成 videos/pushup_5reps.mp4 的脚本（学生不用运行）。

pushup_2s.mp4 是一次完整的俯卧撑（撑起 → 下降 → 撑起，2 秒，来自 Ultralytics 示例视频）。
把它首尾相接重复 5 次，模拟连续做 5 个俯卧撑，供实验 7-3 计数用。
"""
from pathlib import Path

import cv2

HERE = Path(__file__).parent / "videos"
cap = cv2.VideoCapture(str(HERE / "pushup_2s.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
h, w = frames[0].shape[:2]
out = cv2.VideoWriter(str(HERE / "pushup_5reps.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for _ in range(5):
    for f in frames:
        out.write(f)
out.release()
print("完成：", len(frames) * 5, "帧")
