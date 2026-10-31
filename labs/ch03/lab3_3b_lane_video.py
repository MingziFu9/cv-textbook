# 实验 3-3（续）把车道线检测用到一段行车视频上
# %% 第 1 步：把上面的步骤写成一个函数
import cv2
import numpy as np

def find_lanes(img):
    h, w = img.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150)
    mask = np.zeros_like(edges)
    top = int(h * 0.6)
    cv2.fillPoly(mask, np.array([[(int(w * 0.1), h), (int(w * 0.45), top), (int(w * 0.55), top), (int(w * 0.95), h)]]), 255)
    lines = cv2.HoughLinesP(cv2.bitwise_and(edges, mask), 2, np.pi / 180, 30, minLineLength=20, maxLineGap=100)
    found = 0
    for x1, y1, x2, y2 in ([] if lines is None else lines.reshape(-1, 4)):
        if abs((y2 - y1) / (x2 - x1 + 1e-6)) > 0.5:          # 只要够斜的线
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4); found += 1
    return img, found

# %% 第 2 步：一帧一帧地读视频、处理、写出
cap = cv2.VideoCapture("../data/videos/lane_8s.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("lab3_3_lanes.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
frames = empty = 0
while True:
    ok, frame = cap.read()                       # 读一帧；读完了 ok 就是 False
    if not ok:
        break
    frame, n = find_lanes(frame)
    frames += 1
    if n == 0: empty += 1
    out.write(frame)
cap.release(); out.release()
print(f"共 {frames} 帧（{w}×{h}，{fps:.0f} 帧/秒），其中 {empty} 帧没找到车道线")
