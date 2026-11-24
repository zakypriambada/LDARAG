"""
Menjalankan Baseline Dense Retrieval menggunakan ChromaDB untuk korpus pilot (1.000 artikel, 60 kueri).
Menggunakan model paraphrase-multilingual-MiniLM-L12-v2 dengan metrik Cosine HNSW.
Mengevaluasi dua varian: ef_search bawaan dan ef_search=100.
Menghasilkan skema JSONL identik dengan baseline_results_sample_1000.jsonl.
"""

import json
from pathlib import Path
import time

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


# =========================================================
# KONFIGURASI JALUR BERKAS
# =========================================================
ROOT_DIR = Path(__file__).resolve().parent

QUERY_PATH = ROOT_DIR / "data" / "processed" / "queries_pilot_fixed.jsonl"
EMBEDDINGS_PATH = ROOT_DIR / "models" / "chunk_embeddings_sample_1000.npy"
METADATA_PATH = ROOT_DIR / "models" / "chunk_metadata_sample_1000.jsonl"
CHROMA_DIR = ROOT_DIR / "chroma_db" / "sample_1000"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

COLLECTION_NAME = "pilot_chunks_sample_1000"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 20
BATCH_SIZE = 32
TOLERANSI_VERIFIKASI = 1e-4


def baca_jsonl(path: Path):
    """Membaca berkas JSONL dan mengembalikan daftar dictionary."""
    data = []
    with path.open("r", encoding="utf-8") as file:
        for nomor_baris, baris in enumerate(file, start=1):
            baris = baris.strip()
            if baris:
                try:
                    data.append(json.loads(baris))
                except json.JSONDecodeError as err:
                    raise ValueError(
                        f"Gagal mem-parsing baris {nomor_baris} pada {path}: {err}"
                    )
    return data


def tulis_jsonl(path: Path, rows: list):
    """Menulis daftar dictionary ke berkas JSONL secara atomik."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def simpan_json(data: dict, path: Path):
    """Menyimpan dictionary ke berkas JSON terformat rapi."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def hitung_metrik(hasil_kueri: list, latensi_list_ms: list, nama_varian: str, ef_search_val: int):
    """Menghitung metrik IR standar (Hit@K, MRR, Recall@20) dan latensi pencarian."""
    total = len(hasil_kueri)
    if total == 0:
        raise ValueError("Tidak ada kueri untuk dievaluasi.")

    hits = {k: 0 for k in (1, 3, 5, 20)}
    reciprocal_ranks = []

    for item in hasil_kueri:
        rank = item["source_article_rank"]
        if rank is None:
            reciprocal_ranks.append(0.0)
            continue

        reciprocal_ranks.append(1.0 / rank)
        for k in hits:
            if rank <= k:
                hits[k] += 1

    metrik = {
        f"Hit@{k}": hits[k] / total
        for k in hits
    }
    metrik["MRR"] = float(np.mean(reciprocal_ranks))
    metrik["Recall@20"] = hits[20] / total
    metrik["total_queries"] = total
    metrik["successful_queries_top20"] = hits[20]

    # Statistik latensi kueri
    metrik["variant"] = nama_varian
    metrik["ef_search"] = ef_search_val
    metrik["latency_mean_ms"] = float(np.mean(latensi_list_ms))
    metrik["latency_std_ms"] = float(np.std(latensi_list_ms))
    metrik["latency_median_ms"] = float(np.median(latensi_list_ms))
    metrik["latency_min_ms"] = float(np.min(latensi_list_ms))
    metrik["latency_max_ms"] = float(np.max(latensi_list_ms))

    return metrik


def verifikasi_skor_semantik(collection, query_embeddings, queries, metadata_rows, chunk_embeddings):
    """
    Memverifikasi pada beberapa sampel bahwa semantic_score (1 - jarak cosine)
    identik dengan perkalian titik numpy (toleransi 1e-4).
    """
    print("\n--- Verifikasi Nilai Skor Semantik (Chroma vs NumPy) ---")
    chunk_id_ke_index = {meta["chunk_id"]: idx for idx, meta in enumerate(metadata_rows)}

    # Periksa 5 kueri pertama
    jumlah_sampel = min(5, len(queries))
    for q_idx in range(jumlah_sampel):
        q_vec = query_embeddings[q_idx]
        res = collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=1,
            include=["distances"],
        )
        top_chunk_id = res["ids"][0][0]
        top_dist = res["distances"][0][0]
        semantic_score = float(1.0 - top_dist)

        # Hitung perkalian titik numpy untuk chunk yang sama
        row_idx = chunk_id_ke_index[top_chunk_id]
        numpy_dot = float(np.dot(chunk_embeddings[row_idx], q_vec))

        selisih = abs(semantic_score - numpy_dot)
        if selisih > TOLERANSI_VERIFIKASI:
            raise ValueError(
                f"Verifikasi skor gagal pada kueri {q_idx + 1} (chunk: {top_chunk_id})! "
                f"Skor Chroma: {semantic_score:.6f}, Dot NumPy: {numpy_dot:.6f}, "
                f"Selisih: {selisih:.6e} > toleransi {TOLERANSI_VERIFIKASI}"
            )
        print(
            f"  [Sampel {q_idx + 1}] Chunk '{top_chunk_id[:35]}...': "
            f"Chroma={semantic_score:.6f}, NumPy={numpy_dot:.6f}, Selisih={selisih:.2e} (VALID)"
        )

    print("Verifikasi selesai: Semua sampel skor ekuivalen dalam toleransi 1e-4.\n")


def jalankan_retrieval(collection, queries, query_embeddings, nama_varian: str):
    """Menjalankan query ke koleksi ChromaDB untuk seluruh kueri dan mencatat latensi."""
    hasil_kueri = []
    latensi_list_ms = []

    print(f"Menjalankan retrieval untuk varian '{nama_varian}' (60 kueri)...")

    for q_idx, q_item in enumerate(queries):
        source_id = str(q_item["source_article_id"])
        q_vec = query_embeddings[q_idx]

        # Ukur waktu pencarian murni per kueri
        t_mulai = time.perf_counter()
        res = collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        t_selesai = time.perf_counter()

        durasi_ms = (t_selesai - t_mulai) * 1000.0
        latensi_list_ms.append(durasi_ms)

        ids = res["ids"][0]
        distances = res["distances"][0]
        documents = res["documents"][0]
        metadatas = res["metadatas"][0]

        candidates = []
        for chunk_rank, (cid, dist, doc_text, meta) in enumerate(
            zip(ids, distances, documents, metadatas), start=1
        ):
            candidates.append({
                "chunk_rank": chunk_rank,
                "chunk_id": str(cid),
                "article_id": str(meta["article_id"]),
                "semantic_score": float(1.0 - dist),
                "text": str(doc_text),
            })

        # Hitung peringkat artikel sumber berdasarkan kemunculan pertama (deduplikasi artikel)
        unique_article_ids = []
        for cand in candidates:
            art_id = cand["article_id"]
            if art_id not in unique_article_ids:
                unique_article_ids.append(art_id)

        source_article_rank = (
            unique_article_ids.index(source_id) + 1
            if source_id in unique_article_ids
            else None
        )

        hasil_kueri.append({
            "query_id": q_idx + 1,
            "query": q_item["query"],
            "source_article_id": source_id,
            "source_article_rank": source_article_rank,
            "source_found_in_top20_chunks": any(
                cand["article_id"] == source_id for cand in candidates
            ),
            "candidates": candidates,
        })

    return hasil_kueri, latensi_list_ms


def main():
    # 1. Validasi berkas masukan
    for path in (QUERY_PATH, EMBEDDINGS_PATH, METADATA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Berkas tidak ditemukan: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("BASELINE CHROMADB PILOT RETRIEVAL (SAMPLE 1.000)")
    print("==================================================")

    # 2. Muat data kueri, metadata, dan embedding numpy (untuk verifikasi)
    queries = baca_jsonl(QUERY_PATH)
    metadata_rows = baca_jsonl(METADATA_PATH)
    chunk_embeddings = np.load(EMBEDDINGS_PATH)

    print(f"Jumlah kueri pilot  : {len(queries)}")
    print(f"Jumlah chunk korpus : {len(metadata_rows)}")

    # 3. Muat model embedding dan lakukan encoding kueri
    print(f"\nMemuat model SentenceTransformer: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    query_texts = [q["query"] for q in queries]
    print(f"Meng-encode {len(query_texts)} kueri (batch_size={BATCH_SIZE}, normalize_embeddings=True)...")
    query_embeddings = model.encode(
        query_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 4. Hubungkan ke ChromaDB PersistentClient
    print(f"\nMenghubungkan ke ChromaDB: {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Koleksi ditemukan: '{COLLECTION_NAME}' ({collection.count()} item)")

    # 5. Verifikasi skor semantik terhadap numpy
    verifikasi_skor_semantik(
        collection, query_embeddings, queries, metadata_rows, chunk_embeddings
    )

    # 6. Definisi varian pengujian
    varian_list = [
        {
            "nama": "efdefault",
            "label": "ef_search bawaan (default ChromaDB 1.5.9)",
            "ef_search": collection.configuration.get("hnsw", {}).get("ef_search", 100),
            "results_path": OUTPUT_DIR / "baseline_results_sample_1000_chroma_efdefault.jsonl",
            "metrics_path": OUTPUT_DIR / "baseline_metrics_sample_1000_chroma_efdefault.json",
        },
        {
            "nama": "ef100",
            "label": "ef_search=100 (eksplisit)",
            "ef_search": 100,
            "results_path": OUTPUT_DIR / "baseline_results_sample_1000_chroma_ef100.jsonl",
            "metrics_path": OUTPUT_DIR / "baseline_metrics_sample_1000_chroma_ef100.json",
        },
    ]

    hasil_semua_varian = {}

    for var in varian_list:
        print(f"\n==================================================")
        print(f"MEMPROSES VARIAN: {var['label']}")
        print(f"==================================================")

        # Atur konfigurasi ef_search jika varian kedua
        if var["nama"] == "ef100":
            try:
                collection.modify(configuration={"hnsw": {"ef_search": 100}})
            except Exception as e:
                print(f"Info konfigurasi: {e}")

        hasil_kueri, latensi_list_ms = jalankan_retrieval(
            collection, queries, query_embeddings, var["nama"]
        )

        # Simpan detail JSONL
        tulis_jsonl(var["results_path"], hasil_kueri)
        print(f"Detail hasil tersimpan: {var['results_path']}")

        # Hitung metrik
        metrik = hitung_metrik(
            hasil_kueri, latensi_list_ms, var["nama"], var["ef_search"]
        )
        simpan_json(metrik, var["metrics_path"])
        print(f"Metrik tersimpan      : {var['metrics_path']}")

        hasil_semua_varian[var["nama"]] = metrik

        # Tampilkan ringkasan metrik
        print("\n--- Ringkasan Metrik Retrieval ---")
        for k in ("Hit@1", "Hit@3", "Hit@5", "Hit@20", "MRR"):
            print(f"  {k:10s}: {metrik[k]:.4f}")
        print(f"  Rata-rata latensi: {metrik['latency_mean_ms']:.2f} ms (+/- {metrik['latency_std_ms']:.2f} ms)")

    print("\n==================================================")
    print("LANGKAH 2 SELESAI")
    print("==================================================")


if __name__ == "__main__":
    main()

