"""
Demo: Language-Aware Hard Negative Mining
==========================================
Shows side-by-side retrieval comparison (Baseline vs Ours) for sample Flickr30K images.
Saves each comparison as a PNG and optionally stitches into a demo video (demo.mp4).

Run:  python3 demo.py
      python3 demo.py --video       (also create demo.mp4)
      python3 demo.py --query 42    (use image #42 as the query)
"""

import os
import io
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(BASE_DIR, "eval_cache")
OUT_DIR    = os.path.join(BASE_DIR, "demo_output")
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 9.0
BETA  = 1.0
TAU   = 0.5
GAMMA = 0.4
TOP_K = 5

# Interesting query indices (images where linguistic reweighting changes the ranking)
DEFAULT_QUERIES = [3, 17, 42, 88, 150, 203, 315, 427]

# ── Colours ───────────────────────────────────────────────────────────────────
BG_COLOR      = (245, 245, 245)
HEADER_COLOR  = (46,  46,  46)
CORRECT_COLOR = (30, 120,  40)   # green  — correct match
WRONG_COLOR   = (180, 30,  30)   # red    — wrong top-1
OURS_COLOR    = (199, 91,  18)   # UTD orange
BASE_COLOR    = (80,  80,  80)
WHITE         = (255, 255, 255)
LIGHT_BLUE    = (232, 244, 255)
LIGHT_ORANGE  = (255, 243, 224)


def load_font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def wrap_text(text, max_chars=60):
    """Wrap text at word boundaries."""
    words = text.split()
    lines, current = [], ""
    for w in words:
        if len(current) + len(w) + 1 <= max_chars:
            current = (current + " " + w).strip()
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def draw_result_card(draw, x, y, w, rank, caption, is_correct, is_top1, gt_caption,
                     font_small, font_med, font_bold, bg):
    """Draw one retrieved caption card."""
    h = 95
    draw.rectangle([x, y, x+w, y+h], fill=bg, outline=(200,200,200), width=1)

    # Rank badge
    badge_col = CORRECT_COLOR if is_correct else (150, 150, 150)
    draw.rectangle([x+6, y+8, x+34, y+36], fill=badge_col)
    draw.text((x+13, y+12), f"#{rank}", fill=WHITE, font=font_bold)

    # Caption text (wrapped)
    lines = wrap_text(caption, max_chars=55)
    ty = y + 10
    for line in lines[:3]:
        draw.text((x+42, ty), line, fill=(30,30,30), font=font_small)
        ty += 18

    # Correct / wrong label
    if is_correct:
        draw.text((x+w-85, y+8), "✓ CORRECT", fill=CORRECT_COLOR, font=font_small)
    elif rank == 1 and not is_correct:
        draw.text((x+w-80, y+8), "✗ WRONG", fill=WRONG_COLOR, font=font_small)

    return h + 6


def make_demo_image(query_img, query_caption, base_results, ours_results,
                    query_idx, gt_caption):
    """Create a side-by-side comparison image."""
    IMG_SIZE   = 300
    PANEL_W    = 520
    MARGIN     = 20
    TOTAL_W    = IMG_SIZE + PANEL_W * 2 + MARGIN * 4
    TOTAL_H    = 680

    canvas = Image.new("RGB", (TOTAL_W, TOTAL_H), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    font_title = load_font(22)
    font_bold  = load_font(14)
    font_med   = load_font(13)
    font_small = load_font(12)

    # ── Header ────────────────────────────────────────────────────────────────
    draw.rectangle([0, 0, TOTAL_W, 50], fill=HEADER_COLOR)
    draw.text((20, 12),
              "Language-Aware Hard Negative Mining  |  Flickr30K Retrieval Demo",
              fill=WHITE, font=font_title)

    # ── Query image ───────────────────────────────────────────────────────────
    img_resized = query_img.resize((IMG_SIZE, IMG_SIZE))
    canvas.paste(img_resized, (MARGIN, 65))
    draw.text((MARGIN, 375), f"Query Image #{query_idx}", fill=HEADER_COLOR, font=font_bold)
    # GT caption (wrapped)
    draw.text((MARGIN, 395), "Ground truth caption:", fill=(80,80,80), font=font_small)
    for i, line in enumerate(wrap_text(gt_caption, max_chars=28)):
        draw.text((MARGIN, 413 + i*16), line, fill=CORRECT_COLOR, font=font_small)

    # ── Baseline panel ────────────────────────────────────────────────────────
    bx = IMG_SIZE + MARGIN * 2
    draw.rectangle([bx, 58, bx+PANEL_W, 65+IMG_SIZE+TOP_K*101+30],
                   fill=(250,250,252), outline=(180,180,180), width=1)
    draw.rectangle([bx, 58, bx+PANEL_W, 88], fill=BASE_COLOR)
    draw.text((bx+10, 64), f"Baseline  (β=0)  —  Top {TOP_K} Retrieved Captions",
              fill=WHITE, font=font_bold)

    cy = 95
    for rank, (cap, is_correct) in enumerate(base_results, 1):
        bg = (235, 255, 235) if is_correct else WHITE
        h = draw_result_card(draw, bx+6, cy, PANEL_W-12, rank, cap, is_correct,
                             rank == 1, gt_caption, font_small, font_med, font_bold, bg)
        cy += h

    # ── Ours panel ────────────────────────────────────────────────────────────
    ox = IMG_SIZE + PANEL_W + MARGIN * 3
    draw.rectangle([ox, 58, ox+PANEL_W, 65+IMG_SIZE+TOP_K*101+30],
                   fill=(255, 250, 244), outline=(199,91,18), width=2)
    draw.rectangle([ox, 58, ox+PANEL_W, 88], fill=OURS_COLOR)
    draw.text((ox+10, 64), f"Ours  (β=1)  —  Top {TOP_K} Retrieved Captions",
              fill=WHITE, font=font_bold)

    cy = 95
    for rank, (cap, is_correct) in enumerate(ours_results, 1):
        bg = (235, 255, 235) if is_correct else LIGHT_ORANGE
        h = draw_result_card(draw, ox+6, cy, PANEL_W-12, rank, cap, is_correct,
                             rank == 1, gt_caption, font_small, font_med, font_bold, bg)
        cy += h

    # ── Footer ────────────────────────────────────────────────────────────────
    draw.rectangle([0, TOTAL_H-30, TOTAL_W, TOTAL_H], fill=HEADER_COLOR)
    draw.text((20, TOTAL_H-22),
              "CS 6384 Computer Vision  |  Prajapati · Korake · Anis  |  UTD 2025",
              fill=(180,180,180), font=font_small)

    return canvas


def get_top_k(sim_row, captions, k, gt_idx):
    sorted_idx = np.argsort(-sim_row)
    results = []
    for idx in sorted_idx[:k]:
        results.append((captions[idx], idx == gt_idx))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  action="store_true", help="Also produce demo.mp4")
    parser.add_argument("--query",  type=int, default=None, help="Single query image index")
    args = parser.parse_args()

    queries = [args.query] if args.query is not None else DEFAULT_QUERIES

    # ── Load cached embeddings ────────────────────────────────────────────────
    print("Loading cached embeddings...")
    saved      = torch.load(os.path.join(CACHE_DIR, "clip_embeddings.pt"),
                            map_location="cpu", weights_only=True)
    img_embeds = saved["img_embeds"].numpy()
    txt_embeds = saved["txt_embeds"].numpy()
    sbert_np   = torch.load(os.path.join(CACHE_DIR, "sbert_embeddings.pt"),
                            map_location="cpu", weights_only=True).numpy()

    sim_embed = img_embeds @ txt_embeds.T   # (N, N)
    sim_ling  = sbert_np @ sbert_np.T       # (N, N)

    # Linguistic penalty matrix
    ling_pen = BETA * GAMMA * np.maximum(0.0, sim_ling - TAU)
    np.fill_diagonal(ling_pen, 0.0)
    sim_ours = sim_embed - ling_pen

    # ── Load dataset (images + captions) ─────────────────────────────────────
    print("Loading Flickr30K test images...")
    dataset  = load_dataset("clip-benchmark/wds_flickr30k", split="test")
    images, captions = [], []
    for row in dataset:
        img_bytes = row["jpg"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB") if isinstance(img_bytes, bytes) \
              else img_bytes.convert("RGB")
        txt = row["txt"]
        if isinstance(txt, bytes): txt = txt.decode("utf-8")
        images.append(img)
        captions.append(txt.strip())

    # ── Generate demo images ──────────────────────────────────────────────────
    out_frames = []
    for qi in queries:
        if qi >= len(images):
            print(f"  Skipping {qi} (out of range)")
            continue
        print(f"  Query #{qi}: {captions[qi][:60]}...")

        base_results = get_top_k(sim_embed[qi], captions, TOP_K, qi)
        ours_results = get_top_k(sim_ours[qi],  captions, TOP_K, qi)

        frame = make_demo_image(
            images[qi], captions[qi],
            base_results, ours_results,
            qi, captions[qi]
        )
        out_path = os.path.join(OUT_DIR, f"demo_query_{qi:04d}.png")
        frame.save(out_path)
        out_frames.append(frame)
        print(f"    Saved: {out_path}")

        base_rank1_correct = base_results[0][1]
        ours_rank1_correct = ours_results[0][1]
        print(f"    Baseline top-1 correct: {base_rank1_correct}  |  "
              f"Ours top-1 correct: {ours_rank1_correct}")

    print(f"\nDone! {len(out_frames)} demo images in: {OUT_DIR}/")

    # ── Optional video ────────────────────────────────────────────────────────
    if args.video and out_frames:
        try:
            import cv2
            h, w = out_frames[0].size[1], out_frames[0].size[0]
            video_path = os.path.join(BASE_DIR, "demo.mp4")
            out_vid = cv2.VideoWriter(video_path,
                                      cv2.VideoWriter_fourcc(*"mp4v"), 0.5, (w, h))
            for frame in out_frames:
                # Hold each frame for 4 seconds at 0.5 fps = 2 frames per image
                arr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                for _ in range(2):
                    out_vid.write(arr)
            out_vid.release()
            print(f"Video saved: {video_path}")
        except ImportError:
            print("opencv-python not installed. Run: pip3 install opencv-python")
            print("Then re-run with --video flag.")


if __name__ == "__main__":
    main()
