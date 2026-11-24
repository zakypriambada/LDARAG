"""
Re-ranking berbasis skor topik LDA pada hasil Dense Retrieval ChromaDB untuk korpus pilot (1.000 artikel, 60 kueri).
Logika identik dengan rerank_sample_1000.py:
- Preprocessing teks kueri sama (penanda portal, lowercase, stopword, Sastrawi).
- Inferensi distribusi topik kueri via LDA (K=20).
- Kesamaan Hellinger (1 - jarak Hellinger).
- Min-Max normalization per kueri pada 20 kandidat.
- Fusi linier skor semantik dan skor topik (alpha = 0.0 s.d. 1.0).
- Deduplikasi artikel sumber dengan tie-break chunk_rank.
- Verifikasi hasil alpha=0.0 identik dengan baseline Chroma.
- Keluaran tersimpan di folder baru: data/processed/rerank_sample_1000_chroma/
"""

import argparse
import json
from pathlib import Path
import re
import time

from gensim.corpora import Dictionary
from gensim.models import LdaModel
import nltk
from nltk.corpus import stopwords
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from penanda_portal import hapus_penanda_portal


# =========================================================
# KONFIGURASI JALUR BERKAS
# =========================================================
ROOT_DIR = Path(__file__).resolve().parent

DEFAULT_BASELINE_PATH = (
    ROOT_DIR / "data" / "processed" / "baseline_results_sample_1000_chroma_ef100.jsonl"
)
CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "chunks_sample_1000.jsonl"
MODEL_PATH = ROOT_DIR / "models" / "lda_model_sample_1000.model"
DICT_PATH = ROOT_DIR / "models" / "lda_dictionary_sample_1000.dict"
CONFIG_PATH = ROOT_DIR / "models" / "lda_config_sample_1000.json"
DIST_PATH = ROOT_DIR / "models" / "topic_distributions_sample_1000.npy"
META_PATH = ROOT_DIR / "models" / "topic_metadata_sample_1000.jsonl"

OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "rerank_sample_1000_chroma"


# =========================================================
# PREPROCESSING TEKS (IDENTIK DENGAN TRAIN_LDA & RERANK LAMA)
# =========================================================
try:
    stop_words = set(stopwords.words("indonesian"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("indonesian"))

stop_words.update({
    "cnn", "indonesia", "jakarta", "sumber",
    "foto", "video", "halaman", "berita",
})

stemmer = StemmerFactory().create_stemmer()


def preprocess_text(text: str):
    """Preprocessing teks kueri: bersihkan penanda portal, regex, stopword, dan stemming."""
    text = hapus_penanda_portal(str(text))
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [
        token for token in tokens
        if token not in stop_words and len(token) > 1
    ]
    return [stemmer.stem(token) for token in tokens]


def read_jsonl(path: Path):
    """Membaca berkas JSONL."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list):
    """Menulis berkas JSONL secara aman."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def query_topic_distribution(query: str, lda_model, dictionary):
    """Menghitung distribusi topik kueri dengan model LDA."""
    tokens = preprocess_text(query)
    bow = dictionary.doc2bow(tokens)

    if not bow:
        return None

    topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
    probs = np.zeros(lda_model.num_topics, dtype=np.float64)

    for topic_id, probability in topics:
        probs[topic_id] = probability

    total = probs.sum()
    if total <= 0:
        return None

    return probs / total


def hellinger_similarity(p, q):
    """Menghitung kesamaan Hellinger: 1 - Hellinger_distance."""
    distance = np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
    return float(np.clip(1.0 - distance, 0.0, 1.0))


def evaluate_rankings(rows: list):
    """Menghitung metrik peringkat artikel sumber (Hit@1, Hit@3, Hit@5, Hit@20, MRR)."""
    hit1 = hit3 = hit5 = hit20 = 0
    reciprocal_ranks = []

    for row in rows:
        relevant_rank = row["source_article_rank"]
        if relevant_rank is not None:
            if relevant_rank <= 1:
                hit1 += 1
            if relevant_rank <= 3:
                hit3 += 1
            if relevant_rank <= 5:
                hit5 += 1
            if relevant_rank <= 20:
                hit20 += 1
            reciprocal_ranks.append(1.0 / relevant_rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(rows)
    return {
        "num_queries": n,
        "Hit@1": hit1 / n if n else 0.0,
        "Hit@3": hit3 / n if n else 0.0,
        "Hit@5": hit5 / n if n else 0.0,
        "Hit@20": hit20 / n if n else 0.0,
        "MRR": float(np.mean(reciprocal_ranks)) if n else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Re-ranking LDA berbasis ChromaDB untuk korpus pilot."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=str(DEFAULT_BASELINE_PATH),
        help="Path ke berkas baseline hasil ChromaDB JSONL.",
    )
    args = parser.parse_args()
    baseline_path = Path(args.baseline)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Berkas baseline tidak ditemukan: {baseline_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("HYBRID LDA RE-RANKING ATAS CHROMADB (PILOT 1.000)")
    print("==================================================")
    print(f"Berkas baseline masukan: {baseline_path}")
    print(f"Direktori keluaran     : {OUTPUT_DIR}")

    # 1. Muat model LDA, kamus, dan distribusi topik yang sudah ada
    print("\n[1/3] Memuat model LDA dan distribusi topik dokumen...")
    lda = LdaModel.load(str(MODEL_PATH))
    dictionary = Dictionary.load(str(DICT_PATH))
    distributions = np.load(DIST_PATH)

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if lda.num_topics != int(config["selected_num_topics_candidate"]):
        raise ValueError("Jumlah topik tidak sesuai konfigurasi model.")

    metadata = read_jsonl(META_PATH)
    chunks = read_jsonl(CHUNKS_PATH)
    baseline = read_jsonl(baseline_path)

    if distributions.shape != (len(metadata), lda.num_topics):
        raise ValueError(f"Shape distribusi tidak sesuai: {distributions.shape}")

    if len(metadata) != len(chunks):
        raise ValueError("Jumlah metadata tidak sama dengan jumlah chunk.")

    topic_by_chunk = {
        row["chunk_id"]: distributions[i]
        for i, row in enumerate(metadata)
    }
    chunk_by_id = {row["chunk_id"]: row for row in chunks}

    print(f"Model LDA               : {lda.num_topics} topik")
    print(f"Jumlah chunk terindeks  : {len(chunks)}")
    print(f"Jumlah kueri baseline   : {len(baseline)}")

    # Validasi bahwa semua 60 kueri memiliki tepat 20 kandidat
    for row in baseline:
        if len(row["candidates"]) != 20:
            raise ValueError(f"Kueri {row['query_id']} tidak memiliki tepat 20 kandidat.")
        for cand in row["candidates"]:
            cid = cand["chunk_id"]
            if cid not in topic_by_chunk:
                raise ValueError(f"Chunk ID '{cid}' tidak ditemukan pada model topik.")

    # 2. Iterasi re-ranking untuk semua bobot alpha 0.0 s.d. 1.0
    print("\n[2/3] Menjalankan pengujian re-ranking untuk alpha 0.0 s.d. 1.0...")
    summary = []
    all_results = []

    # Cache distribusi topik kueri dan waktu profiling
    query_topic_cache = {}
    waktu_inferensi_topik_ms = []

    for row in baseline:
        qid = row["query_id"]
        q_text = row["query"]
        t0 = time.perf_counter()
        qdist = query_topic_distribution(q_text, lda, dictionary)
        t1 = time.perf_counter()
        query_topic_cache[qid] = qdist
        waktu_inferensi_topik_ms.append((t1 - t0) * 1000.0)

    for alpha in np.round(np.arange(0.0, 1.01, 0.1), 1):
        alpha_val = float(alpha)
        results = []
        waktu_fusi_ms = []

        for row in baseline:
            qid = row["query_id"]
            query = row["query"]
            source_id = str(row["source_article_id"])
            qdist = query_topic_cache[qid]

            t1 = time.perf_counter()

            candidates = []
            for candidate in row["candidates"]:
                cid = candidate["chunk_id"]
                topic_score = (
                    hellinger_similarity(qdist, topic_by_chunk[cid])
                    if qdist is not None
                    else 0.0
                )
                candidates.append({
                    **candidate,
                    "topic_score": topic_score,
                    "semantic_score_original": float(candidate["semantic_score"]),
                })

            sem = np.array(
                [c["semantic_score_original"] for c in candidates],
                dtype=np.float64,
            )
            top = np.array(
                [c["topic_score"] for c in candidates],
                dtype=np.float64,
            )

            # Min-Max Normalization pada 20 kandidat per kueri
            sem_range = sem.max() - sem.min()
            top_range = top.max() - top.min()

            sem_norm = (
                (sem - sem.min()) / sem_range
                if sem_range > 1e-12
                else np.zeros_like(sem)
            )
            top_norm = (
                (top - top.min()) / top_range
                if top_range > 1e-12
                else np.zeros_like(top)
            )

            for i, c in enumerate(candidates):
                c["semantic_score_norm"] = float(sem_norm[i])
                c["topic_score_norm"] = float(top_norm[i])
                c["hybrid_score"] = float(
                    (1.0 - alpha_val) * sem_norm[i] + alpha_val * top_norm[i]
                )

            # Pengurutan: skor tertinggi, tie-break berdasarkan chunk_rank asli
            candidates.sort(
                key=lambda c: (
                    -c["hybrid_score"],
                    c["chunk_rank"],
                )
            )

            # Deduplikasi artikel: ambil kemunculan pertama tiap artikel
            seen_articles = set()
            article_rank = 0
            source_rank = None
            ranked_articles = []

            for candidate in candidates:
                cid = candidate["chunk_id"]
                article_id = str(
                    candidate.get("article_id", chunk_by_id[cid]["article_id"])
                )

                if article_id in seen_articles:
                    continue

                seen_articles.add(article_id)
                article_rank += 1

                ranked_articles.append({
                    "rank": article_rank,
                    "article_id": article_id,
                    "chunk_id": cid,
                    "score": candidate["hybrid_score"],
                    "semantic_score": candidate["semantic_score_original"],
                    "topic_score": candidate["topic_score"],
                    "text": candidate.get("text", ""),
                })

                if article_id == source_id and source_rank is None:
                    source_rank = article_rank

            t2 = time.perf_counter()
            waktu_fusi_ms.append((t2 - t1) * 1000.0)

            results.append({
                "query_id": qid,
                "query": query,
                "source_article_id": source_id,
                "source_article_rank": source_rank,
                "source_found_in_top20_chunks": source_rank is not None,
                "a_LDA": alpha_val,
                "ranked_articles": ranked_articles,
            })

        # Verifikasi khusus alpha=0.0: harus identik dengan baseline Chroma
        if alpha_val == 0.0:
            for b_row, r_row in zip(baseline, results):
                if b_row["source_article_rank"] != r_row["source_article_rank"]:
                    raise ValueError(
                        f"Inkonsistensi pada alpha=0.0 kueri #{r_row['query_id']}: "
                        f"Rank baseline={b_row['source_article_rank']} vs Re-rank={r_row['source_article_rank']}"
                    )
            print("  [Verifikasi Alpha=0.0] Hasil 100% IDENTIK dengan baseline Chroma (VALID)")

        metrics = evaluate_rankings(results)
        metrics["a_LDA"] = alpha_val
        metrics["waktu_topik_mean_ms"] = float(np.mean(waktu_inferensi_topik_ms))
        metrics["waktu_fusi_mean_ms"] = float(np.mean(waktu_fusi_ms))
        summary.append(metrics)
        all_results.append({
            "a_LDA": alpha_val,
            "metrics": metrics,
            "results": results,
        })

        output_path = OUTPUT_DIR / f"results_alpha_{alpha_val:.1f}.jsonl"
        write_jsonl(output_path, results)

    # 3. Simpan ringkasan semua bobot ke JSON
    print("\n[3/3] Menyimpan ringkasan evaluasi...")
    with (OUTPUT_DIR / "metrics_all_alpha.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Pilih bobot terbaik berdasarkan MRR; jika seri, Hit@5; jika masih seri, a_LDA terkecil
    best = sorted(
        summary,
        key=lambda m: (m["MRR"], m["Hit@5"], -m["a_LDA"]),
        reverse=True,
    )[0]

    with (OUTPUT_DIR / "best_alpha_sample_1000.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    # Cetak tabel ringkasan
    print("\n=========================================================================")
    print("HASIL RE-RANKING LDA CHROMADB (SAMPLE 1.000)")
    print("=========================================================================")
    print(f"{'a_LDA':<6} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'Hit@20':<8} | {'MRR':<8} | {'Waktu Fusi':<12}")
    print("-" * 73)
    for m in summary:
        print(
            f"{m['a_LDA']:<6.1f} | "
            f"{m['Hit@1']:<8.4f} | "
            f"{m['Hit@3']:<8.4f} | "
            f"{m['Hit@5']:<8.4f} | "
            f"{m['Hit@20']:<8.4f} | "
            f"{m['MRR']:<8.4f} | "
            f"{m['waktu_fusi_mean_ms']:<10.3f} ms"
        )
    print("=========================================================================")
    print(f"Rata-rata waktu inferensi topik kueri (Sastrawi + LDA): {summary[0]['waktu_topik_mean_ms']:.2f} ms")
    print(f"Rata-rata waktu komputasi fusi & sort kandidat        : {best['waktu_fusi_mean_ms']:.3f} ms")
    print(f"\nBobot alpha terbaik terpilih:")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"\nSemua berkas hasil re-ranking tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

