"""实验 7-1 读视频、抽帧、写视频：每秒抽一帧存成图片，再加上文字合成新视频。

在 labs/ch07 目录下运行：python lab7_1_video_io.py
"""
# %% 第 1 步：打开视频，读出基本信息
from pathlib import Path

import cv2

path = "../data/videos/highway_12s.mp4"          # 高速公路行车视频，12 秒  ← 改这里
cap = cv2.VideoCapture(path)                     # 打开视频，得到一个“读视频的工具”
fps = cap.get(cv2.CAP_PROP_FPS)                  # 帧率：每秒多少张
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))       # 一共多少帧
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))       # 宽
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))      # 高
print(f"帧率 {fps}，共 {n} 帧，{w}×{h}，时长 {n / fps:.1f} 秒")

# %% 第 2 步：一帧一帧地读，每秒存一张图片
out_dir = Path("lab7_1_frames")
out_dir.mkdir(exist_ok=True)
i, saved = 0, 0
while True:
    ok, frame = cap.read()                       # 读下一帧；读完了 ok 就是 False
    if not ok:
        break
    if i % int(fps) == 0:                        # 每隔 fps 帧（1 秒）存一张  ← 改这里
        cv2.imwrite(str(out_dir / f"frame_{i:04d}.jpg"), frame)
        saved += 1
    i += 1
cap.release()                                    # 用完关掉
print("读了", i, "帧，存了", saved, "张图片")

# %% 第 3 步：每一帧加上时间和帧号，写成新视频
cap = cv2.VideoCapture(path)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")         # 视频编码格式，mp4 用 mp4v
writer = cv2.VideoWriter("lab7_1_out.mp4", fourcc, fps, (w, h))   # 文件名、编码、帧率、(宽, 高)
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    text = f"t={i / fps:5.2f}s  frame {i}"         # 当前时间和帧号
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    writer.write(frame)                          # 写入一帧
    i += 1
cap.release()
writer.release()                                 # 一定要 release，否则视频文件可能打不开
print("新视频已保存：lab7_1_out.mp4，共", i, "帧")

# %% 第 4 步：快放和慢放——只改写入的帧率
cap = cv2.VideoCapture(path)
fast = cv2.VideoWriter("lab7_1_fast.mp4", fourcc, fps * 2, (w, h))   # 帧率翻倍 → 2 倍速
while True:
    ok, frame = cap.read()
    if not ok:
        break
    fast.write(frame)
cap.release()
fast.release()
check = cv2.VideoCapture("lab7_1_fast.mp4")
print("2 倍速视频：", int(check.get(cv2.CAP_PROP_FRAME_COUNT)), "帧，时长",
      round(check.get(cv2.CAP_PROP_FRAME_COUNT) / check.get(cv2.CAP_PROP_FPS), 1), "秒")
