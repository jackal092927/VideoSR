#!/usr/bin/env python3
"""
sa2va_test.py — minimal test harness for ByteDance Sa2VA-8B on a real video.

What it does (two-step):
  1) KEY OBJECT PROBE (optional): ask Sa2VA (video QA) to name the single key object
     to track given your problem description + video context.
  2) SEGMENTATION: ask Sa2VA to segment that object across the video frames and
     save masks + an overlay preview video.

Works with: https://huggingface.co/ByteDance/Sa2VA-8B (Apache-2.0)
Model expects a list of frame image paths for videos.

Quick start
-----------
# 0) Environment (Linux + recent NVIDIA GPU recommended)
#    Python 3.10+, PyTorch 2.3+ with CUDA 11.8/12.x
#    (Sa2VA-8B loads in ~16–24GB VRAM; BF16 preferred; CPU is not practical.)
#    If FlashAttention install fails, just run with --no-flash-attn

# 1) Install deps
#    pip install "transformers>=4.42" accelerate pillow opencv-python numpy
#    pip install torch --index-url https://download.pytorch.org/whl/cu121   # or cu118 matching your CUDA
#    # (optional) flash-attn for speed:
#    pip install "flash-attn>=2.5.6" --no-build-isolation

# 2) Run (with key-object discovery):
#    python sa2va_test.py \
#       --video /path/to/video.mp4 \
#       --problem "A ball is thrown horizontally from a table; identify the ball and segment it." \
#       --outdir ./sa2va_out
#    # The script will 1) ask for a key object name, 2) segment it.

# 3) Or skip discovery and directly segment an object phrase:
#    python sa2va_test.py --video demo.mp4 --object "the red ball" --outdir ./sa2va_out

Outputs
-------
  outdir/
    frames/                 # sampled frames given to Sa2VA
    key_object.json         # {"object": "..."}  (if discovery step used)
    prediction.txt          # Sa2VA textual answer for segmentation prompt
    masks/frame_%05d.png    # binary masks (255=object)
    overlay.mp4             # mask overlay visualization

Notes
-----
- Sa2VA video API takes a LIST OF FRAME PATHS (uniformly sampled). We extract frames with OpenCV.
- If mask size differs from frame size, we resize masks to match.
- For very long videos, increase --max-frames or tweak --uniform-sample.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


def extract_frames(video_path: Path, frames_dir: Path, max_frames: int = 16, uniform_sample: bool = True) -> list[str]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = list(range(total))
    if 0 < max_frames < total:
        if uniform_sample:
            # uniform pick including endpoints
            step = max(1, (total - 1) // (max_frames - 1))
            sel = [0] + list(range(1, total - 1, step))[1:max_frames - 1] + [total - 1]
            idxs = sel[:max_frames]
        else:
            # early segment
            idxs = list(range(max_frames))
    written = []
    cur = 0
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            # convert BGR->RGB for saving via PIL for consistency
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            out_path = frames_dir / f"frame_{len(written):05d}.png"
            im.save(out_path)
            written.append(str(out_path))
        cur += 1
    cap.release()
    if not written:
        raise RuntimeError("No frames extracted.")
    return written


def load_sa2va(model_id: str, use_flash_attn: bool, dtype_str: str):
    # Choose dtype
    dtype = torch.bfloat16 if dtype_str.lower() in {"bf16","bfloat16"} else (
            torch.float16 if dtype_str.lower() in {"fp16","float16","half"} else torch.float32)
    
    print(f"Loading {model_id} with dtype={dtype} (flash attention disabled)")
    
    # Load config first and disable flash attention
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # Recursively disable flash attention in all configs
    def disable_flash_attn_recursive(cfg):
        if hasattr(cfg, 'use_flash_attn'):
            cfg.use_flash_attn = False
        if hasattr(cfg, 'attn_implementation'):
            cfg.attn_implementation = 'eager'
        if hasattr(cfg, 'vision_config'):
            disable_flash_attn_recursive(cfg.vision_config)
        if hasattr(cfg, 'llm_config'):
            disable_flash_attn_recursive(cfg.llm_config)
        if hasattr(cfg, 'chat_config'):
            disable_flash_attn_recursive(cfg.chat_config)
    
    disable_flash_attn_recursive(config)
    
    model = AutoModel.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",  # Force eager attention to avoid flash attention issues
    ).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def ask_key_object(model, tokenizer, frames: list[str], problem: str) -> str:
    # Keep it strict to get a short noun phrase
    prompt = (
        "<image>Given the following physics/science problem description, "
        "name the single most important physical object to track in the video. "
        "Respond with just a short noun phrase (no punctuation):\n\n"
        f"Problem: {problem}\n"
    )
    input_dict = {
        "video": frames,
        "text": prompt,
        "past_text": "",
        "mask_prompts": None,
        "tokenizer": tokenizer,
    }
    out = model.predict_forward(**input_dict)
    name = (out.get("prediction", "").strip() or "object")
    # strip trailing punctuation/newlines
    name = name.replace("\n", " ").strip().strip(".:")
    return name


def segment_object(model, tokenizer, frames: list[str], object_phrase: str):
    # Ask for both answer text and segmentation masks
    prompt = (
        f"<image>Please segment {object_phrase} throughout the video. "
        "Return masks aligned to each provided frame, without extra commentary."
    )
    input_dict = {
        "video": frames,
        "text": prompt,
        "past_text": "",
        "mask_prompts": None,
        "tokenizer": tokenizer,
    }
    out = model.predict_forward(**input_dict)
    answer = out.get("prediction", "")
    masks = out.get("prediction_masks", None)  # expected list(np.array(n_frames, H, W), ...) or similar
    return answer, masks


def save_masks_and_overlay(frames: list[str], masks, outdir: Path) -> None:
    masks_dir = outdir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Sa2VA typically returns a list with one array of shape (n_frames, H, W)
    if isinstance(masks, (list, tuple)) and len(masks) == 1 and hasattr(masks[0], "shape"):
        arr = masks[0]
    else:
        arr = masks  # try best-effort

    # Normalize to boolean masks per frame
    per_frame = []
    if isinstance(arr, np.ndarray) and arr.ndim == 3:
        # shape [T, H, W]
        for t in range(arr.shape[0]):
            m = (arr[t] > 0).astype(np.uint8) * 255
            per_frame.append(m)
    elif isinstance(arr, list) and all(isinstance(a, np.ndarray) for a in arr):
        for a in arr:
            m = (a > 0).astype(np.uint8) * 255
            per_frame.append(m)
    else:
        raise RuntimeError("Unexpected mask format from Sa2VA. Inspect return_dict['prediction_masks'].")

    # Save individual masks and build overlay
    # Infer frame size
    ex = Image.open(frames[0])
    W, H = ex.size

    # Video writer (MP4)
    overlay_path = outdir / "overlay.mp4"
    vw = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (W, H))

    for i, fpath in enumerate(frames):
        img = np.array(Image.open(fpath).convert("RGB"))
        mask = per_frame[min(i, len(per_frame)-1)]
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        # red overlay where mask=1
        overlay = img.copy()
        overlay[mask > 0, 0] = 255  # emphasize in red channel
        blended = (0.6*img + 0.4*overlay).astype(np.uint8)
        # write files
        Image.fromarray(mask).save(masks_dir / f"mask_{i:05d}.png")
        vw.write(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True, help="Input video file")
    ap.add_argument("--outdir", type=Path, default=Path("./sa2va_out"))
    ap.add_argument("--model", default="ByteDance/Sa2VA-8B")
    ap.add_argument("--object", default=None, help="Optional: skip discovery and directly segment this noun phrase")
    ap.add_argument("--problem", default=None, help="Problem description to drive the key-object discovery")
    ap.add_argument("--max-frames", type=int, default=12, help="Uniformly sample up to N frames from the video")
    ap.add_argument("--no-flash-attn", action="store_true", help="Disable FlashAttention if unavailable")
    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16","float32"], help="Model dtype")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.outdir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames = extract_frames(args.video, frames_dir, max_frames=args.max_frames, uniform_sample=True)

    model, tokenizer = load_sa2va(args.model, use_flash_attn=not args.no_flash_attn, dtype_str=args.dtype)

    # Step 1: discover key object (if not provided)
    key_obj = args.object
    if key_obj is None:
        if not args.problem:
            print("[warn] --object not provided and --problem is empty. Defaulting to 'the main object'.")
            args.problem = "Please identify the main object relevant to the physical phenomenon."
        key_obj = ask_key_object(model, tokenizer, frames, args.problem)
        (args.outdir / "key_object.json").write_text(json.dumps({"object": key_obj}, indent=2))
        print(f"[discovery] Key object suggested by Sa2VA: {key_obj}")

    # Step 2: segmentation
    print(f"[segment] Requesting masks for: {key_obj}")
    answer, masks = segment_object(model, tokenizer, frames, key_obj)
    (args.outdir / "prediction.txt").write_text(answer)
    print(f"Sa2VA answer: {answer}")

    if masks is None:
        raise RuntimeError("Sa2VA did not return 'prediction_masks'. Try adjusting your prompt.")
    save_masks_and_overlay(frames, masks, args.outdir)
    print(f"Done. See: {args.outdir}")


if __name__ == "__main__":
    main()
