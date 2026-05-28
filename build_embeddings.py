import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# -------------------------
# PATHS
# -------------------------

DB_PATH = Path("ticketmaster_music.db")

DOCS_PATH = Path("docs.npy")
EMB_PATH = Path("embeddings.npy")
META_PATH = Path("metadata.npy")

# -------------------------
# LOAD MODEL
# -------------------------

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# CONNECT DB
# -------------------------

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("""
    SELECT
        name,
        artist,
        date,
        city,
        venue,
        genre,
        subgenre,
        description,
        image,
        url
    FROM events
""")

rows = cursor.fetchall()

print(f"ROWS loaded: {len(rows)}")

if len(rows) == 0:
    raise ValueError("❌ No data found in DB (events table empty or wrong path)")

# -------------------------
# BUILD STRUCTURES
# -------------------------

docs = []
metadata = []

def safe(x):
    return "" if x is None else str(x)

for r in rows:

    # full text used for BM25 + embeddings
    doc = " ".join([
        safe(r["name"]),
        safe(r["artist"]),
        safe(r["date"]),
        safe(r["city"]),
        safe(r["venue"]),
        safe(r["genre"]),
        safe(r["subgenre"]),
        safe(r["description"]),
        safe(r["image"]),
        safe(r["url"])
    ]).lower().strip()

    docs.append(doc)

    # keep structured data for UI (IMPORTANT)
    metadata.append({
        "name": safe(r["name"]),
        "artist": safe(r["artist"]),
        "date": safe(r["date"]),
        "city": safe(r["city"]),
        "venue": safe(r["venue"]),
        "genre": safe(r["genre"]),
        "subgenre": safe(r["subgenre"]),
        "description": safe(r["description"]),
        "image": safe(r["image"]) if "image" in r.keys() else "",
        "url": safe(r["url"]) if "url" in r.keys() else ""  
    })


print(f"DOCS built: {len(docs)}")
print(f"META built: {len(metadata)}")

if len(docs) == 0:
    raise ValueError("❌ docs empty after processing")

# -------------------------
# EMBEDDINGS
# -------------------------

print("Computing embeddings... (this can take time)")

embeddings = model.encode(
    docs,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings)

print(f"EMBEDDINGS shape: {embeddings.shape}")

if len(docs) != len(embeddings):
    raise ValueError("❌ docs / embeddings mismatch")

# -------------------------
# SAVE FILES
# -------------------------

np.save(DOCS_PATH, np.array(docs, dtype=object))
np.save(EMB_PATH, embeddings)
np.save(META_PATH, np.array(metadata, dtype=object))

print("✅ Build complete!")
print("→ docs.npy saved")
print("→ embeddings.npy saved")
print("→ metadata.npy saved")
print("🚀 Index ready")