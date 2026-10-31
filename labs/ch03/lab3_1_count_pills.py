# 实验 3-1 数一数有几颗药片：灰度 → 去噪 → 阈值 → 轮廓计数
# %% 第 1 步：读图、转灰度、去噪、二值化（第 2 章学过的四步）
import cv2

img = cv2.imread("../data/images/pills.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)                    # 先去掉木纹的细小纹理
t, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("大津法算出的门槛：", t)

# %% 第 2 步：找轮廓，数一数
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("找到轮廓：", len(contours), "个")

# %% 第 3 步：量一量每个轮廓的面积，去掉太小的“碎屑”
areas = [cv2.contourArea(c) for c in contours]    # 列表：每个轮廓的面积
print("面积：", sorted(int(a) for a in areas))
pills = [c for c in contours if cv2.contourArea(c) > 300]   # ← 改这里：小于 300 的不算
print("面积大于 300 的：", len(pills), "个")

# %% 第 4 步：面积特别大的，可能是两颗挨在一起
one = sorted(cv2.contourArea(c) for c in pills)[len(pills) // 2]   # 面积的中位数：一颗药片大概多大
total = 0
for c in pills:
    n = max(1, round(cv2.contourArea(c) / one))    # 面积是一颗的几倍，就算几颗
    total = total + n
print(f"一颗药片约 {one:.0f} 像素，按面积估算共 {total} 颗")

# %% 第 5 步：画出外接矩形和编号
show = img.copy()
for i, c in enumerate(pills):                     # enumerate 同时给出编号 i 和轮廓 c
    x, y, w, h = cv2.boundingRect(c)              # 外接矩形：左上角 (x, y)、宽 w、高 h
    cv2.rectangle(show, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(show, str(i + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imwrite("lab3_1_pills.jpg", show)
