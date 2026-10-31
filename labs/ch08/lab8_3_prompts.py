"""实验 8-3 提示词练习：用 Stable Diffusion 生成同一个路口的晴天、雨天、夜间、雾天画面，
再改步数、引导强度、随机种子，比较效果。

需要 GPU（显存 4 GB 以上），在云端 Notebook 的 GPU 环境中运行；第一次运行要下载约 4 GB 的模型。
国内网络如果下载不了，先在终端执行：export HF_ENDPOINT=https://hf-mirror.com
编者环境无法下载模型，本实验的运行结果待在云端 GPU 上实测后补充。
不想写代码的同学，可以在通义万相、即梦、可图等国产在线平台上完成同样的对比（见教材）。
"""
# %% 第 1 步：加载模型
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",      # Stable Diffusion 1.5，生成 512×512 图片
    torch_dtype=torch.float16).to("cuda")                # 半精度，省显存

# %% 第 2 步：同一个路口，换四种天气
base = "a city street intersection seen from a car dashboard camera, cars and traffic lights, realistic photo"
negative = "cartoon, painting, blurry, distorted, low quality, text, watermark"   # 反向提示词：不要什么
weathers = {"sunny": "sunny day, clear sky", "rain": "heavy rain, wet road, reflections",
            "night": "at night, street lights, headlights", "fog": "dense fog, low visibility"}
for name, w in weathers.items():
    g = torch.Generator("cuda").manual_seed(42)           # 固定随机种子，只让天气不同  ← 改这里
    img = pipe(f"{base}, {w}", negative_prompt=negative,
               num_inference_steps=25, guidance_scale=7.5, generator=g).images[0]
    img.save(f"lab8_3_{name}.png")
    print("已生成：", name)

# %% 第 3 步：改步数和引导强度，看看有什么变化
prompt = f"{base}, heavy rain, wet road, reflections"
for steps in [5, 15, 30]:                                 # 去噪步数  ← 改这里
    g = torch.Generator("cuda").manual_seed(42)
    pipe(prompt, negative_prompt=negative, num_inference_steps=steps,
         guidance_scale=7.5, generator=g).images[0].save(f"lab8_3_steps{steps}.png")
for cfg in [1.0, 4.0, 12.0]:                              # 引导强度：多听提示词的话  ← 改这里
    g = torch.Generator("cuda").manual_seed(42)
    pipe(prompt, negative_prompt=negative, num_inference_steps=25,
         guidance_scale=cfg, generator=g).images[0].save(f"lab8_3_cfg{cfg:g}.png")

# %% 第 4 步：换随机种子——同样的提示词，每次画得不一样
for seed in [1, 2, 3]:
    g = torch.Generator("cuda").manual_seed(seed)
    pipe(prompt, negative_prompt=negative, num_inference_steps=25,
         guidance_scale=7.5, generator=g).images[0].save(f"lab8_3_seed{seed}.png")
print("全部完成，请把图片填进实验记录表")
