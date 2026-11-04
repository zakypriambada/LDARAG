
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

EMBEDDINGS_PATH = (
    BASE_DIR / "models" / "chunk_embeddings_sample_1000.npy"
)
METADATA_PATH = (
    BASE_DIR / "models" / "chunk_metadata_sample_1000.jsonl"
)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_data():
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embedding tidak ditemukan: {EMBEDDINGS_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata tidak ditemukan: {METADATA_PATH}"
        )

    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Jumlah embedding dan metadata tidak sama."
        )

    if embeddings.shape[1] != 384:
        raise ValueError(
            f"Dimensi embedding tidak sesuai: {embeddings.shape}"
        )

    return embeddings, metadata


def search(query, embeddings, metadata, model, top_k=20):
    if not query.strip():
        raise ValueError("Query tidak boleh kosong.")

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    # Embedding dokumen dan query sudah dinormalisasi.
    # Dot product setara dengan cosine similarity.
    scores = np.asarray(embeddings @ query_embedding)

    k = min(top_k, len(scores))

    # Mengurutkan kandidat dari skor tertinggi.
    best_indices = np.argsort(scores)[::-1][:k]

    results = []

    for rank, index in enumerate(best_indices, start=1):
        results.append({
            "rank": rank,
            "score": float(scores[index]),
            "article_id": metadata[index]["article_id"],
            "chunk_id": metadata[index]["chunk_id"],
            "text": metadata[index]["text"],
        })

    return results


def main():
    print("Memuat embedding dan metadata sampel...")
    embeddings, metadata = load_data()

    print(f"Jumlah chunk : {len(metadata)}")
    print(f"Dimensi      : {embeddings.shape}")

    print(f"\nMemuat model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    query = input("\nMasukkan query pencarian: ").strip()

    results = search(
        query=query,
        embeddings=embeddings,
        metadata=metadata,
        model=model,
        top_k=20,
    )

    output_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "retrieval_sample_1000.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

    print("\n===== HASIL RETRIEVAL TOP-20 =====")

    for result in results:
        print(f"\nPeringkat : {result['rank']}")
        print(f"Skor      : {result['score']:.4f}")
        print(f"Artikel ID: {result['article_id']}")
        print(f"Chunk ID  : {result['chunk_id']}")
        print(f"Teks      : {result['text'][:300]}...")

    print(f"\nHasil disimpan di: {output_path}")


if __name__ == "__main__":
    main()
