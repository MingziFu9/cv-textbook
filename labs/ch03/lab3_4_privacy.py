# 实验 3-4 给行车照片打码：用轮廓规则找车牌，用检测模型找行人，再打马赛克
# %% 第 1 步：写一个打马赛克的函数：先缩小再放大，细节就没了
import cv2
from ultralytics import YOLO

def mosaic(img, x, y, w, h, block=8):
    part = img[y:y + h, x:x + w]
    small = cv2.resize(part, (max(1, w // block), max(1, h // block)))           # 缩小到 1/8
    img[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)  # 再放大

# %% 第 2 步：用“亮 + 扁长方形”的规则找车牌
img = cv2.imread("../data/images/drive_city.jpg")      # ← 改这里：换成 street_crossing.jpg 试试
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)       # 车牌底色亮
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))  # 把字缝补上
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
plates = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if 8 <= h <= 40 and 2.5 <= w / h <= 6 and cv2.contourArea(c) > 0.45 * w * h:  # 高度、宽高比、填充度
        plates.append((x, y, w, h))
print("像车牌的区域：", plates)

# %% 第 3 步：用检测模型找行人，头部大约在框的上 1/5
r = YOLO("yolo26n.pt")(img, classes=[0], verbose=False)[0]     # 只找类别 0：人
heads = []
for x1, y1, x2, y2 in r.boxes.xyxy.int().tolist():
    heads.append((x1, y1, x2 - x1, max(1, (y2 - y1) // 5)))
print("行人头部区域：", heads)

# %% 第 4 步：全部打码并保存
for x, y, w, h in plates + heads:
    mosaic(img, x, y, w, h)
cv2.imwrite("lab3_4_privacy.jpg", img)
print(f"已打码 {len(plates)} 块车牌、{len(heads)} 个行人头部")
