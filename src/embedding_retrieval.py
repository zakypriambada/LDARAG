
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"
OUTPUT_DIR = BASE_DIR / "models"
EMBEDDINGS_PATH = OUTPUT_DIR / "chunk_embeddings.npy"
METADATA_PATH = OUTPUT_DIR / "chunk_metadata.jsonl"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64


def load_chunks():
    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def build_embeddings():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"File chunk tidak ditemukan: {CHUNKS_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()
    if not chunks:
        raise ValueError("Dataset chunk kosong.")

    print(f"Jumlah chunk: {len(chunks)}")
    print(f"Memuat model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print("Mulai menghitung embedding...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(EMBEDDINGS_PATH, embeddings)

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    print("\n===== HASIL EMBEDDING =====")
    print(f"Jumlah chunk     : {len(chunks)}")
    print(f"Bentuk embedding : {embeddings.shape}")
    print(f"File embedding   : {EMBEDDINGS_PATH}")
    print(f"File metadata    : {METADATA_PATH}")


def search(query, top_k=5):
    """Mencari chunk berdasarkan cosine similarity."""
    if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Embedding belum tersedia. Jalankan build_embeddings() dahulu."
        )

    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = [json.loads(line) for line in file if line.strip()]

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    # Karena embedding telah dinormalisasi, dot product setara
    # dengan cosine similarity.
    scores = np.asarray(embeddings @ query_embedding)

    k = min(top_k, len(scores))
    best_indices = np.argsort(scores)[::-1][:k]

    results = []
    for index in best_indices:
        results.append({
            "score": float(scores[index]),
            "article_id": metadata[index]["article_id"],
            "chunk_id": metadata[index]["chunk_id"],
            "text": metadata[index]["text"],
        })

    return results


if __name__ == "__main__":
    build_embeddings()

    print("\n===== UJI PENCARIAN SEMANTIK =====")
    query = "Apa penyebab seorang tokoh harus berhenti dari aktivitasnya?"
    results = search(query, top_k=5)

    for rank, result in enumerate(results, start=1):
        print(f"\nPeringkat {rank}")
        print(f"Skor      : {result['score']:.4f}")
        print(f"Artikel ID: {result['article_id']}")
        print(f"Chunk ID  : {result['chunk_id']}")
        print(f"Teks      : {result['text'][:300]}...")
