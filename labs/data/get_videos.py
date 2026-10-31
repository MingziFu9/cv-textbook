"""下载第 3、7 章用的行车和健身示例视频（共约 36 MB）。

在 labs/data 目录下运行：python get_videos.py
仓库里已经带了两段剪短、缩小的视频（highway_12s.mp4、lane_8s.mp4、people_2s.mp4、pushup_2s.mp4），一般够用；
想用完整的原始视频再运行本脚本。来源与许可见 labs/data/README.md。
"""
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent / "videos"
UDACITY = "https://raw.githubusercontent.com/udacity/"
FILES = {
    "project_video.mp4": UDACITY + "CarND-Vehicle-Detection/master/project_video.mp4",   # 高速公路，50 秒
    "solidWhiteRight.mp4": UDACITY + "CarND-LaneLines-P1/master/test_videos/solidWhiteRight.mp4",
    "solidYellowLeft.mp4": UDACITY + "CarND-LaneLines-P1/master/test_videos/solidYellowLeft.mp4",
    "people_demo.mp4": "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4",
    "pose_demo.mp4": "https://github.com/ultralytics/assets/releases/download/v0.0.0/solution_ci_pose_demo.mp4",
}

if __name__ == "__main__":
    HERE.mkdir(exist_ok=True)
    for name, url in FILES.items():
        path = HERE / name
        if path.exists():
            print("已存在：", name)
            continue
        print("正在下载：", name)
        urllib.request.urlretrieve(url, path)
    print("完成，视频在", HERE)
