"""超分辨率小工具：Real-ESRGAN 的轻量模型 realesr-general-x4v3（放大 4 倍）。

模型结构和 Real-ESRGAN 官方的 SRVGGNetCompact 相同：32 层 3×3 卷积 + 像素重排（PixelShuffle）放大。
学生直接调用 upscale(img) 即可，不需要读懂本文件。
权重文件来自 https://github.com/xinntao/Real-ESRGAN/releases（BSD-3-Clause 许可）。
"""
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn

URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
WEIGHTS = Path(__file__).parent / "realesr-general-x4v3.pth"


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        layers = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
        for _ in range(num_conv):
            layers += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
        layers.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(layers)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + nn.functional.interpolate(x, scale_factor=self.upscale, mode="nearest")


_model = None


def load():
    global _model
    if _model is None:
        if not WEIGHTS.exists():
            print("第一次使用，正在下载超分辨率模型（约 5 MB）……")
            urllib.request.urlretrieve(URL, WEIGHTS)
        _model = SRVGGNetCompact()
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)["params"]
        _model.load_state_dict(state)
        _model.eval()
    return _model


def upscale(img_bgr):
    """输入 OpenCV 读的 BGR 图片，返回放大 4 倍的 BGR 图片。"""
    model = load()
    x = torch.from_numpy(img_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
    with torch.no_grad():
        y = model(x).clamp(0, 1)
    out = (y[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return out[:, :, ::-1].copy()
