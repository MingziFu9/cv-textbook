"""实验 8-4 图生图与局部重绘：把一张晴天行车照片改成雪天；只重画照片中的一块区域。

需要 GPU（显存 4 GB 以上），在云端 Notebook 的 GPU 环境中运行；第一次运行要下载约 4 GB + 4 GB 的模型。
编者环境无法下载模型，本实验的运行结果待在云端 GPU 上实测后补充。
"""
# %% 第 1 步：准备照片（Stable Diffusion 1.5 适合 512 像素左右的图）
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from PIL import Image

photo = Image.open("../data/images/drive_highway.jpg").convert("RGB").resize((768, 432))   # ← 改这里
photo.save("lab8_4_input.png")

# %% 第 2 步：图生图——保留构图，改成雪天
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
prompt = "highway in heavy snow, snow covered road and hills, winter, realistic dashcam photo"
for strength in [0.3, 0.5, 0.7]:                          # 改动幅度：越大越不像原图  ← 改这里
    g = torch.Generator("cuda").manual_seed(42)
    out = pipe(prompt, image=photo, strength=strength, guidance_scale=7.5,
               num_inference_steps=30, generator=g).images[0]
    out.save(f"lab8_4_snow_{strength}.png")
    print("已生成 strength =", strength)

# %% 第 3 步：做一张遮罩——白色是要重画的区域（右下角的黑色轿车）
mask = Image.new("L", photo.size, 0)                      # 全黑 = 都不改
mask.paste(255, (640, 230, 768, 400))                     # 白色矩形 = 要重画的地方  ← 改这里
mask.save("lab8_4_mask.png")

# %% 第 4 步：局部重绘——把轿车换成一辆红色卡车
inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting", torch_dtype=torch.float16).to("cuda")
g = torch.Generator("cuda").manual_seed(42)
out = inpaint("a red pickup truck on the highway, realistic photo", image=photo, mask_image=mask,
              width=768, height=432, num_inference_steps=30, generator=g).images[0]
out.save("lab8_4_inpaint.png")
print("局部重绘完成")
