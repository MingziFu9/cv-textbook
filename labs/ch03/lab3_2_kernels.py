# 实验 3-2 自己写卷积核：改 3×3 的九个数，看图片变模糊、变锐利、只剩边缘
# %% 第 1 步：读一张灰度图，取一小块看看数字
import cv2
import numpy as np

gray = cv2.imread("../data/images/drive_city.jpg", cv2.IMREAD_GRAYSCALE)   # 直接按灰度读
y, x = 200, 640                                      # 车顶边缘附近的一个点  ← 改这里
patch = gray[y - 1:y + 2, x - 1:x + 2]               # 以 (y, x) 为中心的 3×3 小块
print("3×3 小块：\n", patch)

# %% 第 2 步：写一个卷积核，用 filter2D 在整张图上滑一遍
kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)    # ← 改这里：九个数（这是找竖直边缘的核）
out = cv2.filter2D(gray.astype(np.float32), -1, kernel)
print("手算这一格：", float((patch * kernel).sum()), "  filter2D 算的：", out[y, x])

# %% 第 3 步：换几个现成的卷积核，存图对比
kernels = {
    "blur": np.ones((3, 3), np.float32) / 9,                               # 平均：模糊
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32),  # 锐化
    "edge_v": kernel,                                                    # 竖直边缘
    "edge_h": kernel.T,                                                  # 水平边缘（转置）
}
for name, k in kernels.items():
    res = cv2.filter2D(gray.astype(np.float32), -1, k)
    cv2.imwrite(f"lab3_2_{name}.jpg", np.clip(np.abs(res), 0, 255).astype(np.uint8))
    print(f"{name:8s} 输出的平均值 {np.abs(res).mean():6.1f}")
