from typing import List, Tuple, Optional

import os
import json
import argparse
import shutil

import torch
import torch.distributed as dist
from PIL import Image

from diffusers import AutoencoderKLWan, LucyEditPipeline
from diffusers.utils import export_to_video, load_video


def parse_args():
    parser = argparse.ArgumentParser(description="Lucy-Edit: JSON-driven parallel inference")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated results")
    parser.add_argument("--model_id", type=str, default="/projects/D2DCRC/xiangpeng/models/Lucy-Edit-Dev", help="Path or hub id of Lucy-Edit model")
    parser.add_argument("--seed", type=int, default=0, help="Random seed base")
    parser.add_argument("--num_frames", type=int, default=None, help="Total frames to generate; default uses input frame count")
    parser.add_argument("--source_frames", type=int, default=None, help="Number of input frames to load; default uses all")
    parser.add_argument("--height", type=int, default=None, help="Resize height; default uses source height")
    parser.add_argument("--width", type=int, default=None, help="Resize width; default uses source width")
    parser.add_argument("--fps", type=int, default=24, help="FPS for exported videos")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Classifier-free guidance scale")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    return parser.parse_args()


def derive_ground_instruction(edit_instruction_text: str) -> str:
    # Kept for compatibility; not used when directly using original prompt
    s = (edit_instruction_text or "").strip()
    if s.endswith("."):
        s = s[:-1]
    lower = s.lower()
    prefixes = ["remove ", "delete ", "erase ", "eliminate ", "add ", "make ", "ground "]
    for prefix in prefixes:
        if lower.startswith(prefix):
            s = s[len(prefix):]
            break
    return s


def load_video_frames(
    video_path: str,
    source_frames: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Image.Image], int, int]:
    frames = load_video(video_path)
    if source_frames is not None:
        if len(frames) >= source_frames:
            frames = frames[:source_frames]
        else:
            if len(frames) == 0:
                w, h = target_size if target_size else (832, 480)
                frames = [Image.new("RGB", (w, h), (0, 0, 0)) for _ in range(source_frames)]
            else:
                last = frames[-1]
                frames = frames + [last.copy() for _ in range(source_frames - len(frames))]
    if len(frames) == 0:
        w, h = target_size if target_size else (832, 480)
        frames = [Image.new("RGB", (w, h), (0, 0, 0))]
    w, h = frames[0].size
    if target_size is not None and (w, h) != (target_size[0], target_size[1]):
        frames = [im.resize((target_size[0], target_size[1]), resample=Image.BICUBIC) for im in frames]
        w, h = target_size
    return frames, h, w


def save_side_by_side(input_frames: List[Image.Image], output_frames: List[Image.Image], out_path: str, fps: int) -> None:
    t = min(len(input_frames), len(output_frames))
    if t == 0:
        return
    iw, ih = input_frames[0].size
    ow, oh = output_frames[0].size
    if (ow, oh) != (iw, ih):
        out_resized = [im.resize((iw, ih), resample=Image.BICUBIC) for im in output_frames[:t]]
    else:
        out_resized = output_frames[:t]
    in_crop = input_frames[:t]
    combined: List[Image.Image] = []
    for a, b in zip(in_crop, out_resized):
        canvas = Image.new("RGB", (iw + iw, ih))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (iw, 0))
        combined.append(canvas)
    export_to_video(combined, out_path, fps=fps)


def main():
    args = parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"Running parallel Lucy-Edit inference with {world_size} GPUs")
        print(f"Using seed base: {args.seed}")

    with open(args.test_json, "r", encoding="utf-8") as f:
        items_list = json.load(f)

    eval_prompts = {}
    for idx, item in enumerate(items_list):
        fname = f"{item.get('task_type', 'task')}_{item.get('sample_id', idx)}.mp4"
        eval_prompts[fname] = item

    os.makedirs(args.output_dir, exist_ok=True)

    items = list(eval_prompts.items())
    pending_items = []
    for fname, item in items:
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(args.output_dir, f"gen_{base}.mp4")
        if not os.path.exists(out_path):
            pending_items.append((fname, item))

    if rank == 0:
        print(f"Total items: {len(items)}, already generated: {len(items) - len(pending_items)}, pending: {len(pending_items)}")

    items_per_gpu = len(pending_items) // world_size
    start_idx = rank * items_per_gpu
    end_idx = (rank + 1) * items_per_gpu if rank != world_size - 1 else len(pending_items)
    subset_items = pending_items[start_idx:end_idx]

    print(f"[GPU {rank}] Processing {len(subset_items)} items")

    weight_dtype = torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = LucyEditPipeline.from_pretrained(args.model_id, vae=vae, torch_dtype=weight_dtype)
    pipe.to(f"cuda:{rank}")

    generator = torch.Generator(device=f"cuda:{rank}").manual_seed(args.seed + rank)

    for fname, item in subset_items:
        base = os.path.splitext(fname)[0]
        output_video_path = os.path.join(args.output_dir, f"gen_{base}.mp4")
        info_path = os.path.join(args.output_dir, f"gen_{base}_info.txt")
        input_path = os.path.join(args.output_dir, f"gen_{base}_input.mp4")
        compare_path = os.path.join(args.output_dir, f"gen_{base}_compare.mp4")

        print(f"[GPU {rank}] Processing {fname}...")

        video_path = item.get("source_video_path") or item.get("video_path") or item.get("source_path")
        if video_path is None:
            print(f"[GPU {rank}] Missing 'source_video_path' for {fname}; skipping")
            continue

        # Use original prompt directly from JSON
        prompt = item.get("prompt")
        if not prompt:
            # Fallbacks for robustness if 'prompt' is absent
            prompt = item.get("qwen_vl_72b_refined_instruction")
        if not prompt:
            print(f"[GPU {rank}] Missing prompt for {fname}; skipping")
            continue

        target_size = (args.width, args.height) if args.width is not None and args.height is not None else None
        frames, height, width = load_video_frames(video_path, args.source_frames, target_size)
        num_frames = args.num_frames if args.num_frames is not None else len(frames)

        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                video=frames,
                negative_prompt=args.negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            output_frames = result.frames[0]

        export_to_video(frames, input_path, fps=args.fps)
        # Also copy the original source video for scoring/reference
        source_copy_path = os.path.join(args.output_dir, f"gen_{base}_source.mp4")
        try:
            if os.path.exists(video_path):
                shutil.copy2(video_path, source_copy_path)
        except Exception as e:
            print(f"[GPU {rank}] Warning: failed to copy source video: {e}")
        export_to_video(output_frames, output_video_path, fps=args.fps)
        save_side_by_side(frames, output_frames, compare_path, fps=args.fps)

        with open(info_path, "w", encoding="utf-8") as fp:
            fp.write(prompt)

        print(f"[GPU {rank}] Completed {fname}")

    print(f"[GPU {rank}] Finished processing all assigned items")


if __name__ == "__main__":
    main()
