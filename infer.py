from typing import List, Tuple, Optional

import os
import json
import argparse
import shutil
import numpy as np

import torch
import torch.distributed as dist
from PIL import Image

from diffusers import AutoencoderKLWan, LucyEditPipeline
from diffusers.utils import export_to_video, load_video


# Silence HF tokenizers fork-parallelism warning in torchrun
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    def to_uint8(arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr
        if np.issubdtype(arr.dtype, np.floating):
            # Assume [0,1] floats; clip and scale
            arr = np.clip(arr, 0.0, 1.0) * 255.0
            return arr.astype(np.uint8)
        return arr.astype(np.uint8)

    def frames_to_array(frames_any) -> np.ndarray:
        # Returns array in shape (T, H, W, C) uint8
        if isinstance(frames_any, list):
            if len(frames_any) == 0:
                return np.zeros((0, 1, 1, 3), dtype=np.uint8)
            if isinstance(frames_any[0], Image.Image):
                arr = np.stack([np.asarray(f.convert("RGB")) for f in frames_any], axis=0)
                return to_uint8(arr)
            # list of arrays
            arrs = []
            for f in frames_any:
                a = np.asarray(f)
                if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
                    a = a.transpose(1, 2, 0)
                arrs.append(a)
            arr = np.stack(arrs, axis=0)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            return to_uint8(arr)
        # ndarray inputs
        arr = np.asarray(frames_any)
        if arr.ndim == 5:
            # (B, T, H, W, C)
            arr = arr[0]
        elif arr.ndim == 4:
            # Either (T, H, W, C) or (T, C, H, W)
            if arr.shape[-1] not in (1, 3, 4) and arr.shape[1] in (1, 3, 4):
                arr = arr.transpose(0, 2, 3, 1)
        elif arr.ndim == 3:
            # (H, W, C)
            arr = arr[np.newaxis, ...]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return to_uint8(arr)

    in_arr = frames_to_array(input_frames)
    out_arr = frames_to_array(output_frames)

    if in_arr.shape[0] == 0 or out_arr.shape[0] == 0:
        return

    T = min(in_arr.shape[0], out_arr.shape[0])
    in_arr = in_arr[:T]
    out_arr = out_arr[:T]

    # Ensure same spatial size (should already match given we pass height/width to pipeline)
    if in_arr.shape[1:3] != out_arr.shape[1:3]:
        # Resize out_arr to match in_arr using PIL
        H, W = in_arr.shape[1], in_arr.shape[2]
        resized = []
        for i in range(out_arr.shape[0]):
            im = Image.fromarray(out_arr[i])
            im = im.resize((W, H), resample=Image.BICUBIC)
            resized.append(np.asarray(im))
        out_arr = np.stack(resized, axis=0)

    combined = np.concatenate([in_arr, out_arr], axis=2)
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
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", dtype=torch.float32)
    pipe = LucyEditPipeline.from_pretrained(args.model_id, vae=vae, dtype=weight_dtype)
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

        # Load first to get orientation, then resize to fixed targets:
        # landscape -> HxW 480x832; portrait -> HxW 832x480
        frames, height, width = load_video_frames(video_path, args.source_frames, None)
        if width >= height:
            target_w, target_h = 832, 480
        else:
            target_w, target_h = 480, 832
        if (width, height) != (target_w, target_h):
            frames = [im.resize((target_w, target_h), resample=Image.BICUBIC) for im in frames]
            width, height = target_w, target_h
        num_frames = len(frames)

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
    try:
        dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    main()
