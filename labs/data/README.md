# 实验数据说明

全部数据仅用于教学，请遵守原项目的许可，不得用于商业用途。KITTI 数据集的许可为 CC BY-NC-SA 3.0（署名—非商业性使用—相同方式共享）。

| 文件 | 内容 | 规模 | 来源 | 使用章节 |
|---|---|---|---|---|
| `images/drive_highway.jpg` | 高速公路行车照片 | 1280×720 | Udacity 自动驾驶课程视频 project_video.mp4 第 700 帧，MIT 许可 | 第 1～3、6、9 章 |
| `images/lane_white.jpg`、`lane_yellow.jpg` | 白色、黄色车道线 | 960×540 | Udacity CarND-LaneLines-P1 测试视频，MIT 许可 | 第 2、3 章 |
| `images/drive_city.jpg` | 城市街道（车、行人、厢式车） | 1242×375 | KITTI 005356 | 全书 |
| `images/street_crossing.jpg` | 行人过马路 | 1224×370 | KITTI 004747 | 第 5～9 章 |
| `images/drive_night.jpg` | 模拟夜间画面 | 1242×375 | 编者把 drive_city 调暗、偏蓝、加噪声 | 第 2 章 |
| `images/pills.jpg` | 桌面上的 12 颗白色药片 | 640×480 | 编者用程序绘制 | 第 3 章 |
| `images/portrait_blue.jpg` | 蓝底卡通头像 | 360×480 | 编者用程序绘制，不含真人 | 第 2 章 |
| `images/old_photo.jpg`、`old_photo_mask.png` | 泛黄、带划痕的“老照片”和划痕遮罩 | 600×370 | 编者用 KITTI 004139 加工 | 第 8 章 |
| `images/marker_road.jpg` | 路面上平放的 7 号 ArUco 标记码 | 1280×720 | 编者用 drive_highway 合成（`make_marker_scene.py`） | 第 9 章 |
| `videos/highway_12s.mp4` | 高速公路行车视频 | 12 秒 | Udacity project_video.mp4 剪辑，MIT 许可 | 第 7 章 |
| `videos/lane_8s.mp4` | 车道线视频 | 8 秒 | Udacity CarND-LaneLines-P1，MIT 许可 | 第 3 章 |
| `videos/people_2s.mp4` | 行人视频 | 2 秒 | Ultralytics 示例视频 solutions_ci_demo.mp4，AGPL-3.0 | 第 7 章 |
| `videos/pushup_2s.mp4` | 一次完整的俯卧撑 | 2 秒 | Ultralytics 示例视频 solution_ci_pose_demo.mp4，AGPL-3.0 | 第 7 章 |
| `videos/pushup_5reps.mp4` | 连续 5 个俯卧撑 | 10 秒 | 编者把 pushup_2s 重复拼接（`make_pushup_video.py`） | 第 7 章 |
| `road_objects/` | 车辆行人类型小图，5 类（car、van、truck、pedestrian、cyclist） | 训练 799 张、验证 199 张 | 从 KITTI 标注框裁出（`get_kitti_small.py`） | 第 4 章 |
| `kitti_sample/` | 一张 KITTI 图片和它的 YOLO 标注 | 1 张 | KITTI 000380 | 第 5 章 |
| `kitti_small/`（不在仓库中） | 道路目标检测小样本，4 类 | 训练 400 张、验证 100 张 | 运行 `get_kitti_small.py` 从 Ultralytics 下载 KITTI（约 390 MB）后生成 | 第 5 章 |

**脚本**：`get_kitti_small.py`（KITTI 小样本，学生实验 5-3 前运行）、`get_videos.py`（完整原始视频，可选）；`make_images.py`、`make_marker_scene.py`、`make_pushup_video.py` 是编者生成素材用的，学生不用运行。

**引用 KITTI**：GEIGER A, LENZ P, URTASUN R. Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite[C]//2012 IEEE Conference on Computer Vision and Pattern Recognition. Providence: IEEE, 2012: 3354-3361.
