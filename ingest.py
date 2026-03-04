import os
import json
import lancedb
import pyarrow as pa
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ==============================
# 1. KONFIGURATION
# ==============================

DB_PATH = "./data"
JSONL_PATH = "Urteile_bereinigt_test.jsonl"
TABLE_NAME = "judgments"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# 🔥 WICHTIG: Große Chunk-Batches
CHUNK_BATCH_SIZE = 10000
EMBEDDING_BATCH_SIZE = 32 # Reduziert für MPNet (größeres Modell)

EMBEDDING_DIM = 768


# ==============================
# 2. CPU OPTIMIERUNG
# ==============================

torch.set_num_threads(os.cpu_count())


# ==============================
# 3. SCHEMA
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

    print("Lade Modell...")
    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu"
    )

    db = lancedb.connect(DB_PATH)

    # Sicherstellen, dass die Tabelle gelöscht wird, bevor sie neu erstellt wird
    if TABLE_NAME in db.list_tables():
        print(f"Lösche alte Tabelle '{TABLE_NAME}'...")
        db.drop_table(TABLE_NAME)

    table = db.create_table(TABLE_NAME, schema=SCHEMA)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    print("Zähle Zeilen...")
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    print("Starte Ingest...")

    chunk_buffer = []

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        with tqdm(total=total_lines, desc="Urteile indiziert") as pbar:

            for line in f:
                try:
                    doc = json.loads(line)

                    content = doc.get("content", "")
                    if not content:
                        pbar.update(1)
                        continue

                    chunks = text_splitter.split_text(content)

                    for chunk in chunks:
                        chunk_buffer.append({
                            "text": chunk,
                            "metadata": { # Nest metadata here
                                "file_number": str(doc.get("file_number", "Unbekannt")),
                                "date": str(doc.get("date", "Unbekannt")),
                                "court_name": str(doc.get("court", "Unbekannt")),
                            }
                        })

                    # 🔥 Wenn genug Chunks gesammelt → Embedding rechnen
                    if len(chunk_buffer) >= CHUNK_BATCH_SIZE:

                        texts = [c["text"] for c in chunk_buffer]

                        embeddings = model.encode(
                            texts,
                            batch_size=EMBEDDING_BATCH_SIZE,
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )

                        data_to_insert = []

                        for i, chunk in enumerate(chunk_buffer):
                            vector = np.asarray(
                                embeddings[i], dtype="float32"
                            )
                            chunk["vector"] = vector.tolist()
                            data_to_insert.append(chunk)

                        table.add(data_to_insert)

                        chunk_buffer = []

                    pbar.update(1)

                except Exception as e:
                    print(f"Fehler beim Parsen einer Zeile: {e}")
                    pbar.update(1)

    # ==============================
    # 5. REST FLUSH
    # ==============================

    if chunk_buffer:
        print("Verarbeite Rest-Chunks...")

        texts = [c["text"] for c in chunk_buffer]

        embeddings = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        data_to_insert = []

        for i, chunk in enumerate(chunk_buffer):
            vector = np.asarray(
                embeddings[i], dtype="float32"
            )
            chunk["vector"] = vector.tolist()
            data_to_insert.append(chunk)

        table.add(data_to_insert)

    print("Erstelle Volltext-Index für Stichwortsuche...")
    # In neueren LanceDB Versionen nutzen wir den Standard-Tokenizer explizit
    table.create_fts_index("text", replace=True)

    print("\nFertig! Datenbank ist bereit.")