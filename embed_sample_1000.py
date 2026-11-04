
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = (
    BASE_DIR / "data" / "processed" / "chunks_sample_1000.jsonl"
)
OUTPUT_DIR = BASE_DIR / "models"

EMBEDDINGS_PATH = OUTPUT_DIR / "chunk_embeddings_sample_1000.npy"
METADATA_PATH = OUTPUT_DIR / "chunk_metadata_sample_1000.jsonl"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64


def load_chunks():
    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid pada baris {line_number}"
                ) from exc

            if not chunk.get("chunk_id") or not chunk.get("article_id"):
                raise ValueError(
                    f"ID chunk atau artikel tidak tersedia pada baris {line_number}"
                )

            if not chunk.get("text", "").strip():
                raise ValueError(
                    f"Teks chunk kosong pada baris {line_number}"
                )

            chunks.append(chunk)

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

    print(f"Jumlah chunk : {len(chunks)}")
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

    # Pastikan jumlah embedding sesuai jumlah metadata.
    if len(embeddings) != len(chunks):
        raise ValueError(
            "Jumlah embedding tidak sama dengan jumlah chunk."
        )

    # Simpan ke nama file khusus sampel 1.000 artikel.
    np.save(EMBEDDINGS_PATH, embeddings)

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("\n===== HASIL EMBEDDING =====")
    print(f"Jumlah chunk     : {len(chunks)}")
    print(f"Bentuk embedding : {embeddings.shape}")
    print(f"Tipe data        : {embeddings.dtype}")
    print(f"File embedding   : {EMBEDDINGS_PATH}")
    print(f"File metadata    : {METADATA_PATH}")


if __name__ == "__main__":
    build_embeddings()
