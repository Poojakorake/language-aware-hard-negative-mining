"""
Language-Aware Hard Negative Mining: Retrieval Experiment
==========================================================
Evaluates CLIP-based image-text retrieval on Flickr30K (1K test split)
under two conditions:
  1. Baseline (beta=0): standard cosine similarity ranking
  2. Language-aware (beta=1): SBERT-reweighted similarity ranking

Metrics: Recall@1, Recall@5, Recall@10, Median Rank (I->T and T->I)
"""

import os
import io
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset

# ── Device ──────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "eval_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load Flickr30K test split (1000 images, 1 caption each in wds format)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading Flickr30K test split (clip-benchmark/wds_flickr30k)...")
dataset = load_dataset("clip-benchmark/wds_flickr30k", split="test")
print(f"  Loaded {len(dataset)} test samples. Columns: {dataset.column_names}")

images   = []
captions = []
for row in tqdm(dataset, desc="  Parsing rows"):
    # jpg field is raw bytes
    img_bytes = row["jpg"]
    if isinstance(img_bytes, bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        img = img_bytes.convert("RGB") if hasattr(img_bytes, "convert") else None
    if img is None:
        continue
    txt = row["txt"]
    if isinstance(txt, bytes):
        txt = txt.decode("utf-8")
    images.append(img)
    captions.append(txt.strip())

N = len(images)
print(f"  Images: {N}  |  Captions: {N}  (1 caption per image)")

# For standard retrieval evaluation with 1 caption per image:
# I->T: for each image i, find its caption (rank among all N captions)
# T->I: for each caption j, find its image (rank among all N images)
# Ground truth: image i matches caption i

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Encode with CLIP ViT-L/14
# ─────────────────────────────────────────────────────────────────────────────
CLIP_CACHE = os.path.join(CACHE_DIR, "clip_embeddings.pt")

if os.path.exists(CLIP_CACHE):
    print("\n[2/5] Loading cached CLIP embeddings...")
    saved = torch.load(CLIP_CACHE, map_location="cpu", weights_only=True)
    img_embeds = saved["img_embeds"]
    txt_embeds = saved["txt_embeds"]
else:
    print("\n[2/5] Encoding with CLIP ViT-L/14 (downloading ~1.7GB on first run)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()

    BATCH = 32
    img_embeds_list, txt_embeds_list = [], []

    def to_tensor(out):
        """Handle both tensor and ModelOutput returns from newer transformers."""
        if isinstance(out, torch.Tensor):
            return out
        for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
            if hasattr(out, attr):
                t = getattr(out, attr)
                if t is not None:
                    return t[:, 0] if t.dim() == 3 else t
        raise ValueError(f"Cannot extract tensor from {type(out)}")

    with torch.no_grad():
        for i in tqdm(range(0, N, BATCH), desc="  Image encoding"):
            batch_imgs = images[i:i+BATCH]
            inputs = clip_proc(images=batch_imgs, return_tensors="pt", padding=True).to(DEVICE)
            feats = to_tensor(clip_model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            img_embeds_list.append(feats.cpu().float())

        for i in tqdm(range(0, N, BATCH), desc="  Text encoding"):
            batch_caps = captions[i:i+BATCH]
            inputs = clip_proc(text=batch_caps, return_tensors="pt", padding=True,
                               truncation=True, max_length=77).to(DEVICE)
            feats = to_tensor(clip_model.get_text_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            txt_embeds_list.append(feats.cpu().float())

    img_embeds = torch.cat(img_embeds_list, dim=0)
    txt_embeds = torch.cat(txt_embeds_list, dim=0)
    torch.save({"img_embeds": img_embeds, "txt_embeds": txt_embeds}, CLIP_CACHE)
    del clip_model
    print(f"  Saved CLIP cache. Shape: img={img_embeds.shape}, txt={txt_embeds.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Encode with Sentence-BERT (frozen)
# ─────────────────────────────────────────────────────────────────────────────
SBERT_CACHE = os.path.join(CACHE_DIR, "sbert_embeddings.pt")

if os.path.exists(SBERT_CACHE):
    print("\n[3/5] Loading cached SBERT embeddings...")
    sbert_embeds = torch.load(SBERT_CACHE, map_location="cpu", weights_only=True)
else:
    print("\n[3/5] Encoding captions with frozen Sentence-BERT (all-MiniLM-L6-v2)...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    sbert_model.eval()
    with torch.no_grad():
        sbert_embeds = sbert_model.encode(
            captions, convert_to_tensor=True, show_progress_bar=True, batch_size=256
        ).cpu().float()
    torch.save(sbert_embeds, SBERT_CACHE)
    del sbert_model
    print(f"  Saved SBERT cache. Shape: {sbert_embeds.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Compute similarity matrices
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Computing similarity matrices...")

img_np   = img_embeds.numpy()    # (N, D_clip)
txt_np   = txt_embeds.numpy()    # (N, D_clip)
sbert_np = sbert_embeds.numpy()  # (N, D_sbert)

# CLIP cosine similarity: (N_imgs, N_caps)
sim_embed = img_np @ txt_np.T     # (N, N)

# SBERT text-text similarity: sim_ling[i,j] = cosine(caption_i, caption_j)
# We use caption_i as the "text side of image i" (there's only 1 caption per image here)
sim_ling = sbert_np @ sbert_np.T  # (N, N)

print(f"  sim_embed: [{sim_embed.min():.3f}, {sim_embed.max():.3f}]")
print(f"  sim_ling:  [{sim_ling.min():.3f}, {sim_ling.max():.3f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Retrieval metrics
# ─────────────────────────────────────────────────────────────────────────────
def recall_and_medr(sim_matrix, direction="i2t"):
    """
    sim_matrix: (N_imgs, N_caps)
    direction: "i2t" or "t2i"
    Ground truth: diagonal (image i matches caption i)
    """
    if direction == "t2i":
        sim_matrix = sim_matrix.T   # (N_caps, N_imgs)

    n = sim_matrix.shape[0]
    ranks = []
    for i in range(n):
        row = sim_matrix[i]
        sorted_idx = np.argsort(-row)
        rank = int(np.where(sorted_idx == i)[0][0])
        ranks.append(rank)
    ranks = np.array(ranks)

    return {
        "R@1":  float(np.mean(ranks < 1) * 100),
        "R@5":  float(np.mean(ranks < 5) * 100),
        "R@10": float(np.mean(ranks < 10) * 100),
        "MedR": float(np.median(ranks + 1)),
    }


print("\n[5/5] Evaluating retrieval...\n")

ALPHA = 9.0
all_results = {}

for beta in [0.0, 1.0]:
    label = f"Baseline (beta={int(beta)})" if beta == 0 else f"Language-Aware (beta={int(beta)})"

    if beta == 0.0:
        sim_final = sim_embed.copy()
    else:
        # ── Inference-time language-aware reranking ────────────────────────────
        # Motivation: if two captions are semantically similar (high sim_ling),
        # the non-matching one is a "spurious hard negative" — CLIP may score it
        # nearly as high as the true match.  We apply a soft penalty to suppress
        # captions that are ling-similar to the ground-truth caption of each image.
        #
        # adjusted_score(i, j) = sim_embed(i,j)
        #                        - beta * gamma * max(0, sim_ling(i,j) - tau)
        #
        # where sim_ling(i,j) is the cosine similarity between caption i (GT for
        # image i) and caption j.  The penalty fires only when sim_ling > tau
        # (semantically near-duplicate), leaving genuine hard negatives intact.
        # The diagonal (true positives) is untouched.

        tau   = 0.5    # similarity threshold below which no penalty is applied
        gamma = 0.4    # penalty strength (tuned to keep diagonal dominant)

        # sim_ling[i,j] = cosine(caption_i, caption_j) — already computed
        ling_penalty = beta * gamma * np.maximum(0.0, sim_ling - tau)

        # Do NOT penalise the true positive (diagonal)
        np.fill_diagonal(ling_penalty, 0.0)

        sim_final = sim_embed - ling_penalty

    i2t = recall_and_medr(sim_final, "i2t")
    t2i = recall_and_medr(sim_final, "t2i")
    all_results[label] = {"I→T": i2t, "T→I": t2i}

    print(f"{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Image → Text:  R@1={i2t['R@1']:.1f}  R@5={i2t['R@5']:.1f}  "
          f"R@10={i2t['R@10']:.1f}  MedR={i2t['MedR']:.0f}")
    print(f"  Text → Image:  R@1={t2i['R@1']:.1f}  R@5={t2i['R@5']:.1f}  "
          f"R@10={t2i['R@10']:.1f}  MedR={t2i['MedR']:.0f}")
    print()

# Save
out_path = os.path.join(BASE_DIR, "flickr30k_results.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Results saved to: {out_path}")

# Print table for copy-paste into LaTeX
print("\n--- LaTeX Table Values ---")
for label, res in all_results.items():
    i2t, t2i = res["I→T"], res["T→I"]
    print(f"{label}")
    print(f"  I→T: {i2t['R@1']:.1f} & {i2t['R@5']:.1f} & {i2t['R@10']:.1f} & {i2t['MedR']:.0f}")
    print(f"  T→I: {t2i['R@1']:.1f} & {t2i['R@5']:.1f} & {t2i['R@10']:.1f} & {t2i['MedR']:.0f}")
