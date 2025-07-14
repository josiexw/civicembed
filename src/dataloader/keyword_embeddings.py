# src/dataloader/keyword_embeddings.py

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json

# === Config ===
TEXT_MODEL = "bert-base-multilingual-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_JSONL = "./data/opendata/opendataswiss_metadata.jsonl"
OUTPUT_PATH = "./data/keyword_vecs.pt"

if __name__ == "__main__":
    # === Load metadata and extract keywords ===
    metadata = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            keywords = entry.get("keywords", [])
            text = ", ".join(keywords) if keywords else ""
            metadata.append((entry["id"], text))

    # === Load BERT ===
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    bert = AutoModel.from_pretrained(TEXT_MODEL).to(DEVICE)
    bert.eval()

    # === Encode keywords ===
    vectors = []
    with torch.no_grad():
        for _, text in tqdm(metadata, desc="Encoding keywords"):
            if not text.strip():
                vectors.append(torch.zeros(bert.config.hidden_size))  # fallback to zero vector
                continue
            tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
            vec = bert(**tokens).last_hidden_state[:, 0].squeeze().cpu()  # CLS token
            vectors.append(vec)

    keyword_vecs = torch.stack(vectors)
    torch.save(keyword_vecs, OUTPUT_PATH)
    print(f"Saved {len(keyword_vecs)} keyword vectors to {OUTPUT_PATH}")
