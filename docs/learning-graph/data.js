// 学习图谱数据：由编者整理，概念与定义取自附录 D 术语表
// 改完后运行 python tools/graph_text.py，检查数据并更新 index.md 里的文字版
var CATS = ["示例类别一", "示例类别二"];
var CHAPTERS = ["", "走进视觉", "图像处理", "图像特征", "图像分类", "目标检测", "图像分割", "视频分析", "生成式视觉", "应用实战"];
var CHDIRS = ["", "01-intro", "02-image-processing", "03-features", "04-classification", "05-detection", "06-segmentation", "07-video", "08-generative", "09-applications"];
var NODES = [
  {"id": "示例术语", "ch": 1, "cat": 0, "a": "g-示例术语", "d": "这里是术语表中的定义，鼠标悬停时显示"},
  {"id": "另一个术语", "ch": 1, "cat": 1, "a": "g-另一个术语", "d": "以示例术语为基础的概念"}
];
var EDGES = [["示例术语", "另一个术语"]];
