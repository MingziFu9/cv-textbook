"""实验 9-3 综合项目模板：把你的视觉小程序做成网页。

在 labs/ch09 目录下运行：python lab9_3_project_template.py
模板默认做“数一数图里有几个人、几辆车”，能直接运行。做项目时只需要改标了“← 改这里”的地方。
"""
# %% 第 1 步：加载模型（按你的项目选一个或几个）
import cv2
import gradio as gr
from ultralytics import YOLO

model = YOLO("yolo26n.pt")      # ← 改这里：检测 yolo26n.pt、姿态 yolo26n-pose.pt、分割 yolo26n-seg.pt、你自己训练的 best.pt
PROJECT = "我的视觉小工具"         # ← 改这里：项目名称
NOTICE = "本工具仅供学习演示，结果可能有误，请以实际情况为准。"


# %% 第 2 步：核心函数——输入一张图，返回一张图和一段文字
def analyze(img_rgb, conf=0.25):
    if img_rgb is None:
        return None, "请先上传一张图片"
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    r = model(img, conf=conf, verbose=False)[0]
    names = [r.names[int(c)] for c in r.boxes.cls]
    # ← 改这里：把你的项目规则写在这里（越线计数、角度计数、换底色……）
    text = f"人 {names.count('person')} 个，车 {names.count('car')} 辆"
    out = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)
    return out, text + "\n" + NOTICE


# %% 第 3 步：做成网页
demo = gr.Interface(
    fn=analyze,
    inputs=[gr.Image(type="numpy", label="上传图片"),
            gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="置信度门槛")],   # ← 改这里：你需要的参数
    outputs=[gr.Image(label="结果"), gr.Textbox(label="说明", lines=4)],
    title=PROJECT,
    examples=[["../data/images/street_crossing.jpg", 0.25]],                  # ← 改这里：你的示例图片
)
if __name__ == "__main__":
    demo.launch()
