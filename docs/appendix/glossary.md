---
title: 附录 D 术语表
description: 本书主要术语的中英文名称、简明定义和例子，按章节顺序排列
---

# 附录 D 术语表

本表按章节顺序列出本书的主要术语，每个术语给出英文名称、简明定义和例子。正文中带虚线下划线的术语，把鼠标停在上面就能看到定义，点击可以跳到本表。

通用术语（如“过拟合”“精确率”）的译名和定义以本系列教材共用的《系列公共术语表》为准，各书保持一致。

## 第 1 章 走进计算机视觉

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-计算机视觉"></span>计算机视觉 | Computer Vision, CV | 让计算机从图片和视频中“看懂”内容（有什么、在哪里、在做什么）的技术 | 货车摄像头发现车道偏离并报警 |
| <span id="g-像素"></span>像素 | Pixel | 组成数字图片的最小方格，每个像素只有一种颜色 | 把照片放大后看到的一个个小方块 |
| <span id="g-分辨率"></span>分辨率 | Resolution | 图片横向和纵向各有多少个像素 | 行车视频画面 1280×720 |
| <span id="g-RGB"></span>RGB | Red Green Blue | 用红、绿、蓝三个 0～255 的数表示一种颜色的方法 | 黄色是红 255、绿 255、蓝 0 |
| <span id="g-通道"></span>通道 | Channel | 图片中同一种颜色分量组成的一张数字表，彩色图有 3 个通道 | OpenCV 读入的图片按蓝、绿、红三个通道存放 |
| <span id="g-灰度图"></span>灰度图 | Grayscale Image | 每个像素只有一个 0～255 的亮度值的图片 | 找车道线前先把彩色画面转成灰度图 |
| <span id="g-图像分类"></span>图像分类 | Image Classification | 给整张图片一个类别和把握的任务，回答“是什么” | 判断一张小图是轿车还是卡车 |
| <span id="g-目标检测"></span>目标检测 | Object Detection | 找出图片中有哪些物体、各在什么位置的任务，每个物体用一个框加类别和置信度表示 | 在路口照片中找出 12 个行人和 1 辆车 |
| <span id="g-图像分割"></span>图像分割 | Image Segmentation | 判断图片中每个像素属于哪个物体或类别，得到精确轮廓的任务 | 把画面中能开的路面一个像素一个像素地标出来 |
| <span id="g-目标跟踪"></span>目标跟踪 | Object Tracking | 在视频的连续画面中给每个目标编号，并保持同一目标的编号不变的任务 | 给前方每辆车发一个“号码牌”，盯住它的位置变化 |
| <span id="g-图像生成"></span>图像生成 | Image Generation | 让模型按要求画出一张新图片的任务 | 输入“雨夜的城市路口”生成一张图 |
| <span id="g-姿态估计"></span>姿态估计 | Pose Estimation | 找出人体关节（关键点）位置的任务，连起来像一个“火柴人” | 找出行人的头、肩、肘、膝等 17 个点 |
| <span id="g-驾驶自动化分级"></span>驾驶自动化分级 | Driving Automation Levels | 按“谁在开车”把驾驶自动化分为 0～5 级的标准，我国标准为 GB/T 40429—2021 | 0～2 级是驾驶辅助，人始终负责开车 |
| <span id="g-高级驾驶辅助系统"></span>高级驾驶辅助系统 | Advanced Driver Assistance System, ADAS | 用摄像头、雷达等感知周围情况，提醒或帮助驾驶员的系统，属于驾驶自动化 0～2 级 | 车道偏离报警、前车碰撞报警、自适应巡航 |
| <span id="g-Notebook"></span>Notebook | Notebook | 网页里的编程环境，代码分成单元格逐个运行，结果直接显示在下面 | 在魔搭社区新建 Notebook 运行实验 1-1 |
| <span id="g-OpenCV"></span>OpenCV | Open Source Computer Vision Library | 开源的计算机视觉工具库，在 Python 中写作 `cv2`，用于读写、处理图片和视频 | `cv2.imread("a.jpg")` 读入一张图片 |
| <span id="g-预训练模型"></span>预训练模型 | Pre-trained Model | 别人已经用大量数据训练好、下载后可直接使用或继续训练的模型 | 第一次运行时自动下载的 `yolo26n.pt` |
| <span id="g-多模态大模型"></span>多模态大模型 | Multimodal Large Model | 能同时理解图片、文字等多种信息的大模型，可以看图回答问题 | 把行车照片发给通义千问，问“前面有几辆车” |

## 第 2 章 图像处理基础：用 OpenCV 修图

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-图像处理"></span>图像处理 | Image Processing | 对图片做裁剪、缩放、调色、去噪等加工，输出仍是一张图片 | 把夜间行车画面调亮 |
| <span id="g-感兴趣区域"></span>感兴趣区域 | Region of Interest, ROI | 只对图片中有用的一块区域做处理，其余部分不管 | 找车道线时只看画面下半部分的路面 |
| <span id="g-插值"></span>插值 | Interpolation | 缩放图片时，根据周围已有像素估算新像素值的方法 | 放大照片时用双线性插值让边缘更平滑 |
| <span id="g-颜色空间"></span>颜色空间 | Color Space | 用几个数描述颜色的一套方法，常见的有 RGB、灰度、HSV | 把照片从 BGR 换到 HSV 再找黄色 |
| <span id="g-HSV"></span>HSV | Hue Saturation Value | 用色调、饱和度、亮度三个数描述颜色的方法，更接近人说颜色的方式；OpenCV 中 H 取 0～180 | 黄色车道线的 H 约为 20～30 |
| <span id="g-掩膜"></span>掩膜 | Mask | 和原图一样大的黑白图，白色（255）表示选中的像素，黑色（0）表示没选中 | cv2.inRange 找出的蓝色背景 |
| <span id="g-对比度"></span>对比度 | Contrast | 图片中亮的地方和暗的地方相差多少，相差越大越“鲜明” | 雾天照片对比度低，看起来发灰 |
| <span id="g-直方图"></span>直方图 | Histogram | 统计图片中每个亮度值各有多少个像素的柱状图，用来判断图片偏亮还是偏暗 | 夜间画面的直方图挤在左边 |
| <span id="g-直方图均衡化"></span>直方图均衡化 | Histogram Equalization | 把挤在一起的亮度值拉开、铺满 0～255 的方法，能让偏暗或发灰的图片看得更清 | cv2.equalizeHist 把夜间画面的平均亮度从 24 拉到 93 |
| <span id="g-噪声"></span>噪声 | Noise | 图片中不属于真实画面、随机出现的亮点、暗点或颗粒 | 夜里拍照时画面上的彩色小颗粒 |
| <span id="g-滤波"></span>滤波 | Filtering | 用每个像素周围一小块（窗口）里的像素重新算出它的值，常用来去噪或模糊 | 用 3×3 窗口做中值滤波去掉噪点 |
| <span id="g-中值滤波"></span>中值滤波 | Median Filter | 把窗口里的像素值从小到大排好，取正中间那个作为新值的滤波方法，特别适合去除椒盐噪声 | cv2.medianBlur(img, 3) |
| <span id="g-高斯滤波"></span>高斯滤波 | Gaussian Filter | 按“离中心越近权重越大”给窗口里的像素加权平均的滤波方法 | cv2.GaussianBlur(img, (5, 5), 0) |
| <span id="g-阈值"></span>阈值 | Threshold | 用来划分两类像素的分界值（门槛） | 亮度大于 127 的算白，否则算黑 |
| <span id="g-二值化"></span>二值化 | Binarization | 按阈值把图片变成只有黑（0）、白（255）两种值的图片 | 把药片照片变成白药片、黑桌面 |
| <span id="g-图像预处理"></span>图像预处理 | Image Preprocessing | 在把图片交给模型或算法之前，统一尺寸、调整亮度、去噪等准备工作 | YOLO 先把照片缩放到 640 像素再识别 |
| <span id="g-函数"></span>函数 | Function | 把一段做好某件事的代码打包起来、起个名字，以后用名字就能反复调用 | def process(path): 处理一张图片 |
| <span id="g-循环"></span>循环 | Loop | 让一段代码对一组东西逐个重复执行的写法，Python 中常用 for | for f in files: 依次处理每张照片 |
| <span id="g-列表"></span>列表 | List | 用方括号把一组东西按顺序括起来的 Python 数据，可以用下标（从 0 开始）取出其中的元素 | ["a.jpg", "b.jpg"]；contours[0] 是第一个轮廓 |

## 第 3 章 图像特征：从边缘到卷积

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-特征"></span>特征 | Feature | 图片中能帮助区分物体的线索，如颜色、边缘、形状、角点、纹理 | 车道线的“亮暗分界 + 斜直线”就是它的特征 |
| <span id="g-手工特征"></span>手工特征 | Hand-crafted Feature | 由人设计规则、用公式提取的特征，看得懂、不需要训练，但规则难以写全 | 用宽高比 2.5～6 的亮长方形找车牌 |
| <span id="g-边缘"></span>边缘 | Edge | 图片中亮度变化剧烈的地方，通常对应物体的边界 | 车道线和路面之间的亮暗分界线 |
| <span id="g-边缘检测"></span>边缘检测 | Edge Detection | 找出图片中边缘像素的方法，最常用的是 Canny 算法（去噪、算变化、细化、双门槛连接） | cv2.Canny(blur, 50, 150) |
| <span id="g-霍夫变换"></span>霍夫变换 | Hough Transform | 让每个边缘点为穿过它的直线投票、票数最多的就是图中直线的方法 | 传统车道线检测中用 cv2.HoughLinesP 找出车道线 |
| <span id="g-轮廓"></span>轮廓 | Contour | 二值图中一块区域首尾相连的闭合边界，可以用来计数和量面积 | 用 cv2.findContours 数出 11 个药片轮廓 |
| <span id="g-外接矩形"></span>外接矩形 | Bounding Rectangle | 刚好把一个轮廓或物体框住的矩形，用左上角坐标和宽、高表示，是检测框的前身 | cv2.boundingRect 给出 (x, y, w, h) |
| <span id="g-卷积"></span>卷积 | Convolution | 卷积核在图片上逐格滑动，每到一处把对应位置的数相乘再全部相加，得到输出的计算 | Sobel 核与车顶边缘的 3×3 小块卷积得到 −309 |
| <span id="g-卷积核"></span>卷积核 | Kernel | 卷积中使用的小表格（如 3×3），里面的数决定卷积起模糊、锐化还是找边缘的作用 | 九个数都是 1/9 的平均核 |
| <span id="g-关键点"></span>关键点 | Keypoint | 图片中很有特点、换个角度或距离也能认出的点，可用于两张图之间的配对 | 窗户的角、招牌上的字；ORB 算法找出的点 |
| <span id="g-模板匹配"></span>模板匹配 | Template Matching | 把一小块样板图在大图上逐格滑动比较，找出最相似位置的方法，只适合找一模一样的东西 | 在屏幕截图中找到某个按钮 |
| <span id="g-人脸检测"></span>人脸检测 | Face Detection | 找出图片中人脸位置的任务，传统方法有 Haar 级联，现在多用深度学习模型 | 数码相机的人脸对焦框 |
| <span id="g-个人信息"></span>个人信息 | Personal Information | 以电子或其他方式记录的、与已识别或可识别的自然人有关的各种信息；人脸等生物识别信息属于敏感个人信息 | 行车视频里路人的脸、其他车辆的车牌 |
| <span id="g-卷积神经网络"></span>卷积神经网络 | Convolutional Neural Network, CNN | 由多层卷积等运算组成、卷积核的数从大量图片中自动学出来的神经网络，是深度学习视觉模型的基础 | AlexNet、ResNet；第一层学出边缘检测器 |
| <span id="g-条件判断"></span>条件判断 | Conditional Statement | 按条件是否成立决定执行哪段代码的写法，Python 中用 if、elif、else | if k < -0.5: 判为左车道线 |

## 第 4 章 图像分类：让模型认出图片

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-置信度"></span>置信度 | Confidence | 模型对自己给出的答案有多大把握，是 0～1 之间的数，常写成百分数 | 厢式车小图被认成 van，置信度 99.2% |
| <span id="g-准确率"></span>准确率 | Accuracy | 预测正确的样本占全部样本的比例 | 验证集 199 张认对 162 张，准确率 81.4% |
| <span id="g-Top-5 准确率"></span>Top-5 准确率 | Top-5 Accuracy | 正确答案出现在模型置信度最高的前 5 个类别中就算认对，以此计算的准确率；只看第一名的叫 Top-1 准确率 | ImageNet 比赛同时报告 Top-1 和 Top-5 准确率 |
| <span id="g-卷积层"></span>卷积层 | Convolutional Layer | 卷积神经网络中用一组卷积核在图上滑动、输出多张特征图的层 | yolo26n-cls 第 0 层用 16 个卷积核输出 112×112×16 |
| <span id="g-池化层"></span>池化层 | Pooling Layer | 把特征图缩小（如每 2×2 取最大值）的层，保留“有没有”，丢掉精确位置 | 2×2 最大池化把 56×56 变成 28×28 |
| <span id="g-参数"></span>参数 | Parameter | 模型中由训练学习得到的数，如卷积核里的每个数和分类层的权重 | yolo26n-cls 约有 153 万个参数 |
| <span id="g-迁移学习"></span>迁移学习 | Transfer Learning | 从在大数据集上预训练好的模型出发，换掉输出层、用少量新数据继续训练的方法 | 从 ImageNet 预训练模型出发训练五类道路目标分类器 |
| <span id="g-训练集"></span>训练集 | Training Set | 用来训练模型的数据，相当于“课本和练习题” | road_objects/train 中的 799 张小图 |
| <span id="g-验证集"></span>验证集 | Validation Set | 训练过程中用来检查效果、挑选最好一版模型的数据，相当于“模拟考试” | road_objects/val 中的 199 张小图 |
| <span id="g-轮次"></span>轮次 | Epoch | 把整个训练集完整使用一遍称为一轮 | 实验 4-3 训练 10 轮 |
| <span id="g-批大小"></span>批大小 | Batch Size | 每次更新参数时所用的一小组样本的数量 | batch=32，799 张图每轮更新 25 次 |
| <span id="g-损失"></span>损失 | Loss | 由损失函数算出的、衡量模型预测与真实答案差距的数，训练就是让它变小 | 训练损失从 1.55 降到 0.35 左右 |
| <span id="g-欠拟合"></span>欠拟合 | Underfitting | 模型太简单或训练不够、没学到规律，训练集和验证集成绩都差的现象 | 从零训练 10 轮验证准确率只有 44.2% |
| <span id="g-过拟合"></span>过拟合 | Overfitting | 模型在训练集上表现好、在新数据上表现差的现象 | 训练损失一直降，验证准确率却开始下降 |
| <span id="g-数据增强"></span>数据增强 | Data Augmentation | 训练时对图片随机翻转、裁剪、调亮度等，让模型每次看到略有不同的图，以减轻过拟合的方法 | Ultralytics 训练时自动做左右翻转 |
| <span id="g-混淆矩阵"></span>混淆矩阵 | Confusion Matrix | 按“真实类别 × 预测类别”交叉统计样本数量的表格，对角线是认对的 | 厢式车被认成卡车 8 张 |
| <span id="g-字典"></span>字典 | Dictionary | 用“键: 值”成对存放数据、按键查找值的 Python 数据 | {0: "car", 1: "van"} |

## 第 5 章 目标检测：找出图中的物体

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-边界框"></span>边界框 | Bounding Box | 框住一个物体的最小矩形，常用左上角和右下角坐标表示 | 行人框 (888,184)-(950,319) |
| <span id="g-交并比"></span>交并比 | Intersection over Union, IoU | 两个框重叠部分的面积除以两个框合起来的面积，取值 0～1，衡量两个框有多吻合 | 预测框比真实框大一圈时 IoU 约 0.39 |
| <span id="g-非极大值抑制"></span>非极大值抑制 | Non-Maximum Suppression, NMS | 对同一物体的多个重叠框，只保留置信度最高的一个、删掉与它交并比过高的其余框的去重方法 | 同一辆车上的 3 个框只留 0.89 那个 |
| <span id="g-两阶段检测器"></span>两阶段检测器 | Two-stage Detector | 先找出可能有物体的候选区域，再对每个区域分类和修正框的检测器 | Faster R-CNN |
| <span id="g-一阶段检测器"></span>一阶段检测器 | One-stage Detector | 看一遍整张图就直接输出所有框和类别的检测器，速度快 | YOLO 系列 |
| <span id="g-置信度门槛"></span>置信度门槛 | Confidence Threshold | 只保留置信度不低于这个值的检测框；调高少误检，调低少漏检 | 门槛 0.25 时留下 8 个框，0.5 时只剩 3 个 |
| <span id="g-误检"></span>误检 | False Positive | 把不是目标的东西报成了目标，或框的位置对不上 | 门槛 0.10 时报出 18 个框，其中 11 个是误检 |
| <span id="g-漏检"></span>漏检 | False Negative | 真实存在的目标没有被检测出来 | 门槛 0.50 时 9 个真实物体漏检了 6 个 |
| <span id="g-精确率"></span>精确率 | Precision | 预测为某类的样本中，真正属于该类的比例（判得准不准） | 报出的 8 个框中 5 个是对的，精确率 62% |
| <span id="g-召回率"></span>召回率 | Recall | 某类的全部样本中，被正确找出来的比例（找得全不全） | 9 个真实物体找出 5 个，召回率 56% |
| <span id="g-平均精度"></span>平均精度 | Average Precision, AP / mAP | 把门槛从高到低滑一遍得到的精确率-召回率曲线下的面积；各类别 AP 的平均叫 mAP | mAP50 表示以交并比 0.5 为标准算出的 mAP |
| <span id="g-数据配置文件"></span>数据配置文件 | Dataset YAML | 告诉 YOLO 训练集、验证集图片在哪里以及类别名称的 YAML 文本文件 | kitti_small.yaml 中的 train、val、names |
| <span id="g-标注工具"></span>标注工具 | Annotation Tool | 在图片上画框、选类别并导出标注文件的软件 | X-AnyLabeling、Label Studio |
| <span id="g-开放词汇检测"></span>开放词汇检测 | Open-vocabulary Detection | 用户输入类别名称的文字，模型就能检测这些类别，无需专门训练的检测方法 | YOLO-World 输入 traffic cone 就能框出锥桶 |

## 第 6 章 图像分割：精确到每个像素

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-语义分割"></span>语义分割 | Semantic Segmentation | 给图片中每个像素一个类别，不区分同类中的第几个 | 把行车照片的像素分成路面、人行道、轿车、天空等 19 类 |
| <span id="g-实例分割"></span>实例分割 | Instance Segmentation | 找出每个可数的物体，并给每个物体单独一张掩膜 | 4 个人、1 辆公交车各一张掩膜 |
| <span id="g-全景分割"></span>全景分割 | Panoptic Segmentation | 每个像素既有类别，人、车等物体还能区分是第几个，相当于语义分割加实例分割 | 路面、天空有类别，车 1、车 2 分得开 |
| <span id="g-羽化"></span>羽化 | Feathering | 把掩膜边缘从 0/1 的硬边模糊成逐渐过渡的值，让抠图边缘更自然 | 用 7×7 高斯滤波羽化人物掩膜 |
| <span id="g-多边形标注"></span>多边形标注 | Polygon Annotation | 沿物体轮廓点一圈点、连成多边形的分割标注方式 | YOLO 分割标注：类别编号后跟各顶点的比例坐标 |
| <span id="g-编码器-解码器"></span>编码器-解码器 | Encoder-Decoder | 编码器负责理解并压缩输入，解码器负责逐步还原出输出的网络结构 | U-Net 先把图缩小到 1/8，再放大回原图大小 |
| <span id="g-平均交并比"></span>平均交并比 | Mean IoU, mIoU | 语义分割中每个类别分别算交并比，再取平均 | 路面、天空、轿车……各算一个交并比后平均 |
| <span id="g-分割一切模型"></span>分割一切模型 | Segment Anything Model, SAM | 根据点、框等提示分割任意物体、不需要专门训练的基础模型，只给掩膜不给类别 | 实验 6-3 用 MobileSAM 点一下分出整辆车 |
| <span id="g-提示"></span>提示 | Prompt | 告诉模型要做什么的输入；在 SAM 中是点、框等，用来指明要分割哪个物体 | points=[[1180, 500]], labels=[1] |
| <span id="g-深度估计"></span>深度估计 | Depth Estimation | 估计图片中每个像素离相机有多远的任务 | 城市街道照片中最近约 5.4 米、最远约 86.6 米 |
| <span id="g-深度图"></span>深度图 | Depth Map | 和原图一样大、每个像素存一个距离值的数字表，常用颜色表示远近 | 近处红、远处蓝 |
| <span id="g-单目深度估计"></span>单目深度估计 | Monocular Depth Estimation | 只用一个摄像头拍的单张照片估计深度，便宜但没有绝对尺子 | yolo26n-depth 估计最近的车约 6.5 米 |
| <span id="g-场景重建"></span>场景重建 | Scene Reconstruction | 根据照片或视频恢复场景的三维结构 | 用手机绕物体拍一圈照片，生成三维模型 |
| <span id="g-布尔索引"></span>布尔索引 | Boolean Indexing | 用 True/False 表放进方括号，只挑出 True 位置的元素 | show[road] = (0, 200, 0) 只把路面涂绿 |

## 第 7 章 视频分析：让模型看懂动态画面

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-帧"></span>帧 | Frame | 视频中的一张图片 | 12 秒的行车视频有 300 帧 |
| <span id="g-帧率"></span>帧率 | Frame Rate, FPS | 每秒播放或拍摄多少帧 | 行车记录仪常用 25 或 30 fps |
| <span id="g-压缩"></span>压缩 | Compression | 利用相邻帧大部分内容相同等特点，把视频文件变小的方法；读出来的每一帧仍是完整图片 | 300 帧原始数据约 198 MB，mp4 文件只有约 4 MB |
| <span id="g-ID"></span>ID | Track ID | 跟踪器给每个目标发的号码牌，在后面的帧里保持不变 | 超车的黑车一直是 ID 2 |
| <span id="g-ID 切换"></span>ID 切换 | ID Switch | 同一个目标在跟踪过程中号码牌变了，常由遮挡或漏检引起，会造成重复计数 | 车从桥下出来被当成新车，发了新号码 |
| <span id="g-越线计数"></span>越线计数 | Line Crossing Count | 目标的号码牌从线的一边移到另一边时计数加一，每个号码只算一次 | 统计路口一共过去了多少辆车 |
| <span id="g-区域闯入"></span>区域闯入 | Zone Intrusion | 目标进入划定的多边形区域时发出提醒 | 行人的脚下点进入车道区域就标红 |
| <span id="g-人体关键点"></span>人体关键点 | Human Keypoints | 姿态估计找出的人体关节等位置，COCO 格式共 17 个 | 鼻子、两肩、两肘、两腕、两髋、两膝、两踝等 |
| <span id="g-动作识别"></span>动作识别 | Action Recognition | 根据一段时间内的画面或关键点变化，判断人在做什么动作 | 数俯卧撑、判断驾驶员是否持续低头 |
| <span id="g-驾驶员监测系统"></span>驾驶员监测系统 | Driver Monitoring System, DMS | 用车内摄像头判断驾驶员是否分心、疲劳并发出提醒的系统 | 连续低头看手机 2 秒以上就提醒 |
| <span id="g-集合"></span>集合 | Set | Python 中元素不重复的容器，用 set() 创建、add() 添加 | 进入过区域的号码牌 {1, 2, 4, 5, 6} |
| <span id="g-端到端"></span>端到端 | End-to-End | 用一个模型直接从输入学到最终输出，中间步骤一起训练，不再分成独立的模块 | 从摄像头视频直接输出行驶路线 |

## 第 8 章 生成式视觉：Stable Diffusion 与图像修复

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-文生图"></span>文生图 | Text-to-Image | 根据一段文字描述生成一张新图片 | 输入“雨天的路口”生成一张雨天路口照片 |
| <span id="g-图生图"></span>图生图 | Image-to-Image | 以一张图片为起点，按文字描述修改风格或内容，同时保留大致构图 | 把晴天的高速公路改成雪天 |
| <span id="g-局部重绘"></span>局部重绘 | Inpainting with Diffusion | 只在遮罩区域内按文字描述重新生成，其他区域不变 | 把右下角的黑色轿车换成红色皮卡 |
| <span id="g-图像修复"></span>图像修复 | Image Inpainting | 根据周围内容填补图片中缺损或被遮挡的区域 | 用 cv2.inpaint 修补老照片的划痕 |
| <span id="g-超分辨率"></span>超分辨率 | Super-Resolution | 把低分辨率图片放大并补出细节，使其看起来更清楚的技术 | Real-ESRGAN 把 100×40 的小图放大到 400×160 |
| <span id="g-生成对抗网络"></span>生成对抗网络 | Generative Adversarial Network, GAN | 由生成器和判别器两个网络互相对抗训练、生成逼真样本的模型 | 生成以假乱真的人脸照片 |
| <span id="g-扩散模型"></span>扩散模型 | Diffusion Model | 训练时学会从带噪声的图中猜出噪声，生成时从纯噪声出发一步步去噪得到新图的生成模型 | Stable Diffusion 用 25 步去噪生成一张图 |
| <span id="g-潜空间"></span>潜空间 | Latent Space | 把图片压缩后的小尺寸表示空间，在其中做扩散可以大大减少计算量 | 512×512 的图压成 64×64 的潜空间表示 |
| <span id="g-提示词"></span>提示词 | Prompt | 给模型的指令和输入，在文生图中是对想要画面的文字描述 | a city street intersection, heavy rain, realistic photo |
| <span id="g-反向提示词"></span>反向提示词 | Negative Prompt | 告诉生成模型不想要哪些内容的文字 | blurry, low quality, text, watermark |
| <span id="g-引导强度"></span>引导强度 | Guidance Scale | 生成时多大程度上听从提示词的参数，太低跑题、太高生硬 | 常用 7～8 |
| <span id="g-随机种子"></span>随机种子 | Random Seed | 决定随机数起点的整数；在扩散模型中决定起始噪声，种子相同则结果可以复现 | manual_seed(42) |
| <span id="g-顶帽运算"></span>顶帽运算 | Top-hat Transform | 原图减去开运算结果，提取比周围亮的细小结构的形态学操作 | 实验 8-2 用它自动找白色划痕 |
| <span id="g-深度合成"></span>深度合成 | Deep Synthesis | 利用深度学习等技术生成或编辑文本、图像、音频、视频的技术 | AI 换脸、AI 配音、AI 写作 |

## 第 9 章 视觉应用实战与综合项目

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
