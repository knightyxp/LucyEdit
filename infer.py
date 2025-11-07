from typing import List

import torch
from PIL import Image

from diffusers import AutoencoderKLWan, LucyEditPipeline
from diffusers.utils import export_to_video, load_video


# Arguments
url = "https://d2drjpuinn46lb.cloudfront.net/painter_original_edit.mp4"
prompt = "Change the outfit to gothic black high-waisted skinny jeans, a cropped black leather moto jacket, and a fitted ribbed crop top; matte leather, silver zippers and studs, tight silhouette, clean seams, soft wrinkles, consistent studio lighting, mid-shot with canvases, natural shadows."
negative_prompt = ""
num_frames = 81
height = 480
width = 832

# Load video
def convert_video(video: List[Image.Image]) -> List[Image.Image]:
    video = load_video(url)[:num_frames]
    video = [video[i].resize((width, height)) for i in range(num_frames)]
    return video

video = load_video("painter_original_edit.mp4")

# Load model
model_id = "/projects/D2DCRC/xiangpeng/models/Lucy-Edit-Dev"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = LucyEditPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Generate video
output = pipe(
    prompt=prompt,
    video=video,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0
).frames[0]

# Export video
export_to_video(output, "painter_gothic_edit_enhance_prompt.mp4", fps=24)
