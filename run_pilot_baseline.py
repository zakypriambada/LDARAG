
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# =========================================================
# KONFIGURASI
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

QUERY_PATH = (
    BASE_DIR / "data" / "processed" / "queries_pilot_fixed.jsonl"
)
EMBEDDINGS_PATH = BASE_DIR / "models" / "chunk_embeddings_sample_1000.npy"
METADATA_PATH = BASE_DIR / "models" / "chunk_metadata_sample_1000.jsonl"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
RESULT_PATH = OUTPUT_DIR / "baseline_results_sample_1000.jsonl"
METRICS_PATH = OUTPUT_DIR / "baseline_metrics_sample_1000.json"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 20
BATCH_SIZE = 32


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def calculate_metrics(query_results):
    total = len(query_results)

    if total == 0:
        raise ValueError("Tidak ada kueri untuk dievaluasi.")

    hits = {k: 0 for k in (1, 3, 5, 20)}
    reciprocal_ranks = []

    for result in query_results:
        rank = result["source_article_rank"]

        if rank is None:
            reciprocal_ranks.append(0.0)
            continue

        reciprocal_ranks.append(1.0 / rank)

        for k in hits:
            if rank <= k:
                hits[k] += 1

    metrics = {
        f"Hit@{k}": hits[k] / total
        for k in hits
    }

    metrics["MRR"] = float(np.mean(reciprocal_ranks))
    metrics["Recall@20"] = hits[20] / total
    metrics["total_queries"] = total
    metrics["successful_queries_top20"] = hits[20]

    return metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Periksa file
    for path in (QUERY_PATH, EMBEDDINGS_PATH, METADATA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

    # 2. Muat kueri, metadata, dan embedding
    queries = read_jsonl(QUERY_PATH)
    metadata = read_jsonl(METADATA_PATH)
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")

    if len(queries) != 30:
        print(f"PERINGATAN: ditemukan {len(queries)} kueri, bukan 30.")

    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            "Jumlah metadata berbeda dari jumlah embedding."
        )

    if embeddings.shape[1] != 384:
        raise ValueError(
            f"Dimensi embedding tidak sesuai: {embeddings.shape}"
        )

    if not all(
        item.get("source_article_id") and item.get("query")
        for item in queries
    ):
        raise ValueError("Ada kueri tanpa ID sumber atau teks pertanyaan.")

    print(f"Jumlah kueri     : {len(queries)}")
    print(f"Jumlah chunk     : {len(metadata)}")
    print(f"Shape embedding  : {embeddings.shape}")
    print(f"Memuat model     : {MODEL_NAME}")

    # 3. Muat model sekali dan encode semua kueri
    model = SentenceTransformer(MODEL_NAME)

    query_texts = [item["query"] for item in queries]

    query_embeddings = model.encode(
        query_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 4. Cari Top-20 chunk dan evaluasi berdasarkan artikel
    all_results = []

    for query_index, query_item in enumerate(queries):
        query_vector = query_embeddings[query_index]

        # Embedding dokumen sudah dinormalisasi.
        scores = np.asarray(embeddings @ query_vector)

        k = min(TOP_K, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]

        candidates = []
        seen_articles = set()
        unique_article_rank = None

        source_id = str(query_item["source_article_id"])

        for chunk_rank, index in enumerate(top_indices, start=1):
            chunk = metadata[int(index)]
            article_id = str(chunk["article_id"])

            # Rank artikel = posisi pertama artikel itu muncul
            # dalam daftar chunk berdasarkan skor semantik.
            if (
                article_id == source_id
                and unique_article_rank is None
            ):
                unique_article_rank = len(seen_articles) + (
                    0 if article_id in seen_articles else 1
                )

            candidates.append({
                "chunk_rank": chunk_rank,
                "chunk_id": chunk["chunk_id"],
                "article_id": article_id,
                "semantic_score": float(scores[index]),
                "text": chunk["text"],
            })

            seen_articles.add(article_id)

        # Hitung ulang peringkat artikel berdasarkan urutan
        # kemunculan pertama, tanpa menghitung artikel berulang.
        unique_article_ids = []
        for candidate in candidates:
            article_id = candidate["article_id"]
            if article_id not in unique_article_ids:
                unique_article_ids.append(article_id)

        source_article_rank = (
            unique_article_ids.index(source_id) + 1
            if source_id in unique_article_ids
            else None
        )

        result = {
            "query_id": query_index + 1,
            "query": query_item["query"],
            "source_article_id": source_id,
            "source_article_rank": source_article_rank,
            "source_found_in_top20_chunks": any(
                candidate["article_id"] == source_id
                for candidate in candidates
            ),
            "candidates": candidates,
        }

        all_results.append(result)

        print(
            f"[{query_index + 1:02d}/{len(queries)}] "
            f"rank artikel sumber: {source_article_rank}"
        )

    # 5. Simpan detail hasil retrieval
    with RESULT_PATH.open("w", encoding="utf-8") as file:
        for result in all_results:
            file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

    # 6. Hitung metrik pilot
    metrics = calculate_metrics(all_results)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    # 7. Tampilkan hasil
    print("\n========== HASIL BASELINE PILOT ==========")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:25s}: {value:.4f}")
        else:
            print(f"{metric:25s}: {value}")

    print(f"\nDetail hasil : {RESULT_PATH}")
    print(f"Metrik       : {METRICS_PATH}")


if __name__ == "__main__":
    main()
