# src/api/location_metadata.py

import json
import pandas as pd
import spacy
from tqdm import tqdm
from langdetect import detect

SPACY_MODELS = {
    'en': spacy.load('en_core_web_sm'),
    'de': spacy.load('de_core_news_sm'),
    'fr': spacy.load('fr_core_news_sm'),
    'it': spacy.load('it_core_news_sm'),
}

def get_spacy_model(text):
    try:
        lang = detect(text)
        return SPACY_MODELS.get(lang, None)
    except:
        return None

def extract_locations(text, nlp):
    if not nlp:
        return set()
    doc = nlp(text)
    locations = set()
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            locations.add(ent.text.strip())
    return locations

INPUT_PATH = './data/opendata/opendataswiss_metadata.jsonl'
OUTPUT_PATH = './data/opendata/opendataswiss_locations.parquet'

results = []

with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        try:
            item = json.loads(line)
            db_id = item.get("id")
            title = item.get("title", "")
            desc = item.get("description", "")
            text = f"{title}. {desc}"
            nlp = get_spacy_model(text)
            locs = extract_locations(text, nlp)
            results.append({"id": db_id, "locations": list(locs)})
        except Exception as e:
            print(f"Error processing line: {e}")

df = pd.DataFrame(results)
df.to_parquet(OUTPUT_PATH, index=False)
