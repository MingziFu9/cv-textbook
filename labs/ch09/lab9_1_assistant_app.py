"""实验 9-1 智能辅助驾驶助手网页：上传一张行车照片，显示车辆行人框、可行驶路面、远近和提醒文字。

在 labs/ch09 目录下运行：python lab9_1_assistant_app.py
运行后在浏览器打开终端里显示的网址（一般是 http://127.0.0.1:7860）。按 Ctrl+C 结束。
"""
# %% 第 1 步：加载前面几章用过的三个模型
import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

det_model = YOLO("yolo26n.pt")          # 第 5 章：检测车和人  ← 可换成第 5 章自己训练的 best.pt
sem_model = YOLO("yolo26n-sem.pt")      # 第 6 章：道路语义分割
depth_model = YOLO("yolo26n-depth.pt")  # 第 6 章：深度估计
WANT = {"person": "行人", "bicycle": "自行车", "car": "车", "motorcycle": "摩托车", "bus": "大客车", "truck": "卡车"}


# %% 第 2 步：核心函数——输入一张图，返回画好的图和提醒文字
def analyze(img_rgb, near_m=10):
    if img_rgb is None:                                        # 还没上传照片
        return None, "请先上传一张行车照片"
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)            # 网页给的是 RGB，OpenCV 用 BGR
    sem = sem_model(img, verbose=False)[0].semantic_mask.data.cpu().numpy()
    depth = depth_model(img, verbose=False)[0].depth.data.cpu().numpy()
    det = det_model(img, verbose=False)[0]
    out = img.copy()
    out[sem == 0] = (0.6 * out[sem == 0] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)  # 路面涂绿
    items = []
    for (x1, y1, x2, y2), c in zip(det.boxes.xyxy.int().tolist(), det.boxes.cls.int().tolist()):
        name = det.names[c]
        if name not in WANT:
            continue
        dist = float(np.median(depth[y1:y2, x1:x2]))           # 框里深度的中位数
        items.append((dist, WANT[name]))
        color = (0, 0, 255) if dist < near_m else (255, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{dist:.0f}m", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), report(sem, items, near_m)


# %% 第 3 步：把结果写成提醒文字
def report(sem, items, near_m):
    h, w = sem.shape
    front = sem[int(h * 0.6):, int(w * 0.35):int(w * 0.65)]   # 车头正前方（实验 6-2）
    lines = [f"画面中路面占 {(sem == 0).mean():.0%}，正前方路面占 {(front == 0).mean():.0%}"]
    counts = {}
    for _, n in items:
        counts[n] = counts.get(n, 0) + 1
    lines.append("找到：" + ("、".join(f"{n} {k} 个" for n, k in counts.items()) if counts else "没有车辆和行人"))
    near = sorted(d for d in items if d[0] < near_m)
    for dist, n in near:
        lines.append(f"⚠ {n}约 {dist:.1f} 米，注意保持距离")
    if not near:
        lines.append(f"{near_m} 米内没有车辆和行人")
    lines.append("（本工具仅供学习演示，距离为模型估计，驾驶时请以实际路况为准）")
    return "\n".join(lines)


# %% 第 4 步：做成网页
demo = gr.Interface(
    fn=analyze,
    inputs=[gr.Image(type="numpy", label="上传行车照片"),
            gr.Slider(5, 30, value=10, step=1, label="提醒距离（米）")],
    outputs=[gr.Image(label="分析结果"), gr.Textbox(label="提醒", lines=6)],
    title="智能辅助驾驶助手",
    description="绿色是可行驶路面；红框表示距离小于提醒距离的车辆或行人，框上数字是估计距离。",
    examples=[["../data/images/drive_city.jpg", 10], ["../data/images/street_crossing.jpg", 10],
              ["../data/images/drive_highway.jpg", 10]],
)
if __name__ == "__main__":
    demo.launch()                       # 想让同一局域网的同学访问：demo.launch(server_name="0.0.0.0")
