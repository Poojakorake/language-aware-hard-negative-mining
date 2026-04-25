import pandas as pd
import json
import os
import ast
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 1. Setup paths (make paths relative to this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "flickr_annotations_30k.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "flickr30k-images")
OUTPUT_JSON = os.path.join(BASE_DIR, "flickr_llave.json")
OUTPUT_EMB = os.path.join(BASE_DIR, "ling_embeddings.pt")

# 2. Device safe model load (CPU fallback)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading frozen Sentence-BERT model on {device}...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
sbert_model.to(device)
sbert_model.eval()

# 3. Read CSV (handle common delimiters)
with open(CSV_PATH, 'r', encoding='utf-8', errors='ignore') as fh:
    head = fh.read(200)
sep = '|' if '|' in head and head.count('|') > head.count(',') else ','
df = pd.read_csv(CSV_PATH, sep=sep, encoding='utf-8', engine='python')
df.columns = [c.strip() for c in df.columns]

llave_data = []
all_captions = []

def parse_raw_field(raw):
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if not isinstance(raw, str):
        return [str(raw).strip()]
    s = raw.strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            parsed = json.loads(s)
            return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            try:
                parsed = ast.literal_eval(s)
                return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                return [s]
    # not a list-like string
    return [s]

print(f"Processing {len(df)} rows and flattening captions...")
for i, row in tqdm(df.iterrows(), total=len(df)):
    # stop after 6000 images (6000 images * ~5 captions = ~30,000 examples)
    if i >= 6000:
        break
    img_name = str(row.get('filename') or row.get('image') or row.get('img_id') or '').strip()
    if not img_name:
        continue
    raw = row.get('raw') if 'raw' in row else row.get('caption') if 'caption' in row else row.get('comment') if 'comment' in row else ''
    captions = parse_raw_field(raw)
    for j, caption in enumerate(captions):
        entry = {
            "id": f"flickr_{i}_{j}",
            "image": os.path.join(IMAGE_DIR, img_name),
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe this image."},
                {"from": "gpt", "value": caption}
            ]
        }
        llave_data.append(entry)
        all_captions.append(caption)

# 4. Save flattened JSON
with open(OUTPUT_JSON, "w", encoding='utf-8') as f:
    json.dump(llave_data, f, indent=2, ensure_ascii=False)

# 5. Pre-compute linguistic embeddings aligned with flattened captions
print("Pre-computing linguistic embeddings (sim_ling)...")
with torch.no_grad():
    embeddings = sbert_model.encode(all_captions, convert_to_tensor=True, show_progress_bar=True)
    # move to CPU before saving to improve portability
    if isinstance(embeddings, torch.Tensor) and embeddings.device.type != 'cpu':
        embeddings = embeddings.cpu()
    torch.save(embeddings, OUTPUT_EMB)

print(f"Done! Created {OUTPUT_JSON} ({len(llave_data)} entries) and {OUTPUT_EMB} ({len(all_captions)} embeddings)")