
import json
from pathlib import Path

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.corpus import stopwords
import nltk
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


ROOT = Path(__file__).resolve().parent

BASELINE_PATH = ROOT / "data/processed/baseline_results_sample_1000.jsonl"
CHUNKS_PATH = ROOT / "data/processed/chunks_sample_1000.jsonl"
MODEL_PATH = ROOT / "models/lda_model_sample_1000.model"
DICT_PATH = ROOT / "models/lda_dictionary_sample_1000.dict"
CONFIG_PATH = ROOT / "models/lda_config_sample_1000.json"
DIST_PATH = ROOT / "models/topic_distributions_sample_1000.npy"
META_PATH = ROOT / "models/topic_metadata_sample_1000.jsonl"

OUTPUT_DIR = ROOT / "data/processed/rerank_sample_1000"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Preprocessing: harus sama dengan train_lda.py
# --------------------------------------------------
try:
    stop_words = set(stopwords.words("indonesian"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("indonesian"))

stop_words.update({
    "cnn", "indonesia", "jakarta", "sumber",
    "foto", "video", "halaman", "berita"
})

stemmer = StemmerFactory().create_stemmer()


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [
        token for token in tokens
        if token not in stop_words and len(token) > 1
    ]
    return [stemmer.stem(token) for token in tokens]


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------
# 2. Muat model dan distribusi topik
# --------------------------------------------------
print("Memuat model dan distribusi topik...")

lda = LdaModel.load(str(MODEL_PATH))
dictionary = Dictionary.load(str(DICT_PATH))
distributions = np.load(DIST_PATH)

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = json.load(f)

if lda.num_topics != int(config["selected_num_topics_candidate"]):
    raise ValueError("Jumlah topik tidak sesuai konfigurasi.")

metadata = read_jsonl(META_PATH)
chunks = read_jsonl(CHUNKS_PATH)
baseline = read_jsonl(BASELINE_PATH)

if distributions.shape != (len(metadata), lda.num_topics):
    raise ValueError(
        f"Shape distribusi tidak sesuai: {distributions.shape}"
    )

if len(metadata) != len(chunks):
    raise ValueError("Jumlah metadata tidak sama dengan jumlah chunk.")

# Pastikan urutan ID chunk sama
for i, (meta, chunk) in enumerate(zip(metadata, chunks)):
    if meta["chunk_id"] != chunk["chunk_id"]:
        raise ValueError(f"ID chunk berbeda pada baris {i}.")

topic_by_chunk = {
    row["chunk_id"]: distributions[i]
    for i, row in enumerate(metadata)
}

chunk_by_id = {row["chunk_id"]: row for row in chunks}

if len(topic_by_chunk) != len(metadata):
    raise ValueError("Ada chunk_id duplikat pada metadata.")

print(f"Model LDA: {lda.num_topics} topik")
print(f"Jumlah chunk: {len(chunks)}")
print(f"Jumlah kueri Baseline: {len(baseline)}")


# --------------------------------------------------
# 3. Hitung distribusi topik untuk kueri
# --------------------------------------------------
def query_topic_distribution(query):
    tokens = preprocess_text(query)
    bow = dictionary.doc2bow(tokens)

    if not bow:
        return None

    topics = lda.get_document_topics(bow, minimum_probability=0.0)
    probs = np.zeros(lda.num_topics, dtype=np.float64)

    for topic_id, probability in topics:
        probs[topic_id] = probability

    total = probs.sum()
    if total <= 0:
        return None

    return probs / total


def hellinger_similarity(p, q):
    # Hellinger distance = norm(sqrt(p)-sqrt(q))/sqrt(2)
    distance = np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
    return float(np.clip(1.0 - distance, 0.0, 1.0))


# --------------------------------------------------
# 4. Evaluasi peringkat artikel sumber
# --------------------------------------------------
def evaluate_rankings(rows):
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


# --------------------------------------------------
# 5. Re-ranking untuk semua bobot a_LDA
# --------------------------------------------------
summary = []
all_results = []

for row in baseline:
    if len(row["candidates"]) != 20:
        raise ValueError(
            f"{row['query_id']} tidak memiliki tepat 20 kandidat."
        )

    for candidate in row["candidates"]:
        cid = candidate["chunk_id"]
        if cid not in topic_by_chunk:
            raise ValueError(f"Chunk ID tidak ditemukan: {cid}")


for alpha in np.round(np.arange(0.0, 1.01, 0.1), 1):
    print(f"Menguji a_LDA = {alpha:.1f}")

    results = []

    for row in baseline:
        query = row["query"]
        source_id = str(row["source_article_id"])
        qdist = query_topic_distribution(query)

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
                "semantic_score_original": float(
                    candidate["semantic_score"]
                ),
            })

        sem = np.array(
            [c["semantic_score_original"] for c in candidates],
            dtype=np.float64,
        )
        top = np.array(
            [c["topic_score"] for c in candidates],
            dtype=np.float64,
        )

        # Min-max normalization pada 20 kandidat untuk kueri yang sama
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
                (1.0 - alpha) * sem_norm[i] + alpha * top_norm[i]
            )

        # Pada alpha=0, pertahankan urutan Baseline saat skor seri.
        if alpha == 0.0:
            candidates.sort(
                key=lambda c: (
                    -c["hybrid_score"],
                    c["chunk_rank"],
                )
            )
        else:
            candidates.sort(
                key=lambda c: (
                    -c["hybrid_score"],
                    c["chunk_rank"],
                )
            )

        # Deduplicasi artikel: ambil chunk dengan skor tertinggi
        seen_articles = set()
        article_rank = 0
        source_rank = None
        ranked_articles = []

        for candidate in candidates:
            cid = candidate["chunk_id"]
            article_id = str(
                candidate.get(
                    "article_id",
                    chunk_by_id[cid]["article_id"]
                )
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

        results.append({
            "query_id": row["query_id"],
            "query": query,
            "source_article_id": source_id,
            "source_article_rank": source_rank,
            "source_found_in_top20_chunks": source_rank is not None,
            "a_LDA": float(alpha),
            "ranked_articles": ranked_articles,
        })

    metrics = evaluate_rankings(results)
    metrics["a_LDA"] = float(alpha)
    summary.append(metrics)
    all_results.append({
        "a_LDA": float(alpha),
        "metrics": metrics,
        "results": results,
    })

    output_path = OUTPUT_DIR / f"results_alpha_{alpha:.1f}.jsonl"
    write_jsonl(output_path, results)

# Simpan ringkasan semua bobot
with (OUTPUT_DIR / "metrics_all_alpha.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# Bobot terbaik berdasarkan MRR; jika seri, Hit@5
best = sorted(
    summary,
    key=lambda m: (m["MRR"], m["Hit@5"], -m["a_LDA"]),
    reverse=True,
)[0]

with (OUTPUT_DIR / "best_alpha_sample_1000.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(best, f, ensure_ascii=False, indent=2)

print("\n=== RINGKASAN EKSPERIMEN ===")
print("a_LDA | Hit@1 | Hit@3 | Hit@5 | Hit@20 | MRR")
for m in summary:
    print(
        f'{m["a_LDA"]:.1f} | '
        f'{m["Hit@1"]:.4f} | '
        f'{m["Hit@3"]:.4f} | '
        f'{m["Hit@5"]:.4f} | '
        f'{m["Hit@20"]:.4f} | '
        f'{m["MRR"]:.4f}'
    )

print("\nBobot terbaik pada data pengembangan:")
print(json.dumps(best, ensure_ascii=False, indent=2))
print(f"\nSemua hasil tersimpan di: {OUTPUT_DIR}")
