import os
import json
import lancedb
import pyarrow as pa
import numpy as np
import torch
import torch_directml
from transformers import AutoTokenizer, AutoModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ==============================
# 1. KONFIGURATION
# ==============================

DB_PATH = "./data"
JSONL_PATH = "Urteile_bereinigt.jsonl"
TABLE_NAME = "judgments"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

CHUNK_SIZE = 1200 
CHUNK_OVERLAP = 200

# 🔥 AGGRESSIVE BATCHING FÜR GPU-AUSLASTUNG
CHUNK_BATCH_SIZE = 10000  # Sammle 10k Chunks vor dem Schreiben
EMBEDDING_BATCH_SIZE = 128 # Große Batchsizer für RX 6750 XT (12GB VRAM)

EMBEDDING_DIM = 768

# ==============================
# 2. GPU SETUP
# ==============================

device = torch_directml.device()
print(f"Nutze Device: {device}")
torch.set_grad_enabled(False)

print(f"Lade Tokenizer und Modell '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_batch(texts):
    # Padding und Truncation auf 384 Tokens (reicht für MPNet meistens aus)
    encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=384, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings.cpu().numpy()


# ==============================
# 3. SCHEMA (Arrow für Speed)
# ==============================

SCHEMA = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    pa.field("text", pa.utf8()),
    pa.field("metadata", pa.struct([ 
        pa.field("file_number", pa.utf8()),
        pa.field("date", pa.utf8()),
        pa.field("court_name", pa.utf8()),
    ])),
])


# ==============================
# 4. MAIN
# ==============================

if __name__ == "__main__":

    db = lancedb.connect(DB_PATH)
    print(f"Initialisiere Tabelle '{TABLE_NAME}'...")
    table = db.create_table(TABLE_NAME, schema=SCHEMA, mode="overwrite")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunk_buffer = []
    
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        with tqdm(desc="Gesamtfortschritt", unit=" chunks") as pbar:

            for line in f:
                try:
                    doc = json.loads(line)
                    content = doc.get("content", "")
                    if not content: continue

                    chunks = text_splitter.split_text(content)

                    for chunk in chunks:
                        chunk_buffer.append({
                            "text": chunk,
                            "metadata": {
                                "file_number": str(doc.get("file_number", "Unbekannt")),
                                "date": str(doc.get("date", "Unbekannt")),
                                "court_name": str(doc.get("court", "Unbekannt")),
                            }
                        })

                        if len(chunk_buffer) >= CHUNK_BATCH_SIZE:
                            # 1. Embeddings berechnen (in großen Batches auf GPU)
                            texts = [c["text"] for c in chunk_buffer]
                            all_embeddings = []
                            
                            for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                                batch_texts = texts[i : i + EMBEDDING_BATCH_SIZE]
                                all_embeddings.append(encode_batch(batch_texts))
                            
                            final_embeddings = np.vstack(all_embeddings)

                            # 2. Daten für Arrow vorbereiten
                            vectors = [final_embeddings[i].tolist() for i in range(len(chunk_buffer))]
                            texts_col = [c["text"] for c in chunk_buffer]
                            metadata_col = [c["metadata"] for c in chunk_buffer]

                            # 3. Via Arrow in die DB (viel schneller!)
                            data_table = pa.Table.from_pydict({
                                "vector": vectors,
                                "text": texts_col,
                                "metadata": metadata_col
                            }, schema=SCHEMA)
                            
                            table.add(data_table)
                            
                            pbar.update(len(chunk_buffer))
                            chunk_buffer = []

                except Exception as e:
                    print(f"\nFehler: {e}")

    # Rest verarbeiten
    if chunk_buffer:
        texts = [c["text"] for c in chunk_buffer]
        all_embeddings = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            all_embeddings.append(encode_batch(texts[i:i+EMBEDDING_BATCH_SIZE]))
        
        final_embeddings = np.vstack(all_embeddings)
        data_table = pa.Table.from_pydict({
            "vector": [final_embeddings[i].tolist() for i in range(len(chunk_buffer))],
            "text": [c["text"] for c in chunk_buffer],
            "metadata": [c["metadata"] for c in chunk_buffer]
        }, schema=SCHEMA)
        table.add(data_table)
        pbar.update(len(chunk_buffer))

    print("\nErstelle FTS Index...")
    table.create_fts_index("text", replace=True)

    print("\nFertig!")