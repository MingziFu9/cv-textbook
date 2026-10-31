"""实验 7-4 驾驶员分心提醒：用人体关键点判断低头、转头，持续 2 秒以上才提醒。

在 labs/ch07 目录下运行：python lab7_4_distraction.py
第 3 步需要你自己录一段模拟开车的视频（坐在椅子上，手机放在正前方录上半身），命名为 my_drive.mp4 放在本文件夹。
"""
# %% 第 1 步：用鼻子和两肩的位置判断头部状态
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import ASSETS

model = YOLO("yolo26n-pose.pt")


def head_state(k):
    """k 是一个人的 17 个关键点，返回 (状态, 低头指标, 转头指标)"""
    nose, ls, rs = np.array(k[0]), np.array(k[5]), np.array(k[6])   # 鼻子、左肩、右肩
    mid = (ls + rs) / 2                           # 两肩中点
    sw = np.linalg.norm(ls - rs)                  # 肩宽，用来把距离变成“几个肩宽”
    if sw < 20:                                   # 肩宽不到 20 像素：肩膀没拍全或人太远
        return "看不清", 0.0, 0.0
    drop = (mid[1] - nose[1]) / sw                # 鼻子比肩膀高多少：越小越像低头
    turn = (nose[0] - mid[0]) / sw                # 鼻子偏离中线多少：绝对值越大越像转头
    if drop < 0.35:                               # 阈值  ← 改这里
        return "低头", drop, turn
    if abs(turn) > 0.4:                           # 阈值  ← 改这里
        return "转头", drop, turn
    return "正常", drop, turn


for name in ["bus.jpg", "zidane.jpg"]:            # Ultralytics 自带的两张示例图片
    r = model(ASSETS / name, verbose=False)[0]
    for k, box in zip(r.keypoints.xy.tolist(), r.boxes.xyxy.int().tolist()):
        if box[2] - box[0] < 100:                 # 太小（太远）的人不看
            continue
        state, drop, turn = head_state(k)
        print(f"{name:<11} 框 {box}  低头指标 {drop:.2f}  转头指标 {turn:+.2f} → {state}")

# %% 第 2 步：持续 2 秒以上才提醒——用一串模拟的逐帧结果试一试
fps = 25
states = ["正常"] * 30 + ["低头"] * 20 + ["正常"] * 10 + ["低头"] * 60 + ["正常"] * 20   # 模拟 5.6 秒
need = 2 * fps                                    # 连续 2 秒 = 50 帧  ← 改这里
run, alerts = 0, []
for i, s in enumerate(states):
    run = run + 1 if s != "正常" else 0           # 连续“不正常”的帧数
    if run == need:                               # 刚好满 2 秒的那一帧发出提醒
        alerts.append(i)
print("模拟", len(states), "帧；在第", alerts, "帧发出提醒，即", [round(i / fps, 1) for i in alerts], "秒")

# %% 第 3 步：处理自己录的视频
video = Path("my_drive.mp4")                      # ← 改这里，换成你的视频文件名
if not video.exists():
    print("没有找到", video, "，请先录一段视频（见本文件开头的说明）")
else:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    writer = cv2.VideoWriter("lab7_4_out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    run, total_alert = 0, 0
    for i, r in enumerate(model(str(video), stream=True, verbose=False)):
        frame = r.plot()                          # 画好关键点的图
        state = "无人"
        if r.keypoints is not None and len(r.boxes) > 0:
            state = head_state(r.keypoints.xy[0].tolist())[0]
        run = run + 1 if state in ("低头", "转头") else 0
        alert = run >= 2 * fps                    # 连续 2 秒
        total_alert += alert
        text = f"{'ALERT! ' if alert else ''}head: {'down' if state == '低头' else 'turn' if state == '转头' else 'ok'}"
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if alert else (0, 200, 0), 2)
        writer.write(frame)
    writer.release()
    print("处理了", i + 1, "帧，提醒状态共", total_alert, "帧，结果在 lab7_4_out.mp4")
