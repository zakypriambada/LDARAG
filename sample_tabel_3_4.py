import json
import random
from pathlib import Path

from check_queries import (
    PENUNJUK,
    kata,
    runtut_terpanjang,
    token_kueri,
)


# =========================================================
# KONFIGURASI
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
QUERY_PATH = BASE_DIR / "data" / "processed" / "queries_final.jsonl"
ARTICLE_PATH = BASE_DIR / "data" / "processed" / "articles_sample_10000.jsonl"

SAMPLE_SIZE = 5
RANDOM_SEED = 42
MIN_UNIQUE_CATEGORIES = 3


def load_articles(path):
    """Membaca artikel korpus dan memetakan ID ke kategori serta ringkasan."""
    articles = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            articles[str(item["id"])] = {
                "category": str(item.get("category", "tidak diketahui")),
                "summary": str(item.get("summary", "")),
            }
    return articles


def load_clean_queries(query_path, articles):
    """Membaca berkas kueri final dan menyaring kueri yang bebas dari daftar audit."""
    clean_queries = []
    skipped = 0

    with open(query_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            sid = str(record.get("source_article_id", ""))
            query = record.get("query")

            if not is_clean_query(sid, query, articles):
                skipped += 1
                continue

            clean_queries.append({
                "source_article_id": sid,
                "category": articles[sid]["category"],
                "query": query.strip(),
            })

    return clean_queries, skipped


def is_clean_query(sid, query, articles):
    """Mengembalikan True jika kueri lolos seluruh kriteria audit."""
    if not query or sid not in articles:
        return False

    query_text = query.strip()
    if not query_text.endswith("?"):
        return False

    if PENUNJUK.search(query_text):
        return False

    summary = articles[sid]["summary"]
    if runtut_terpanjang(token_kueri(query_text), kata(summary)) >= 6:
        return False

    return True


def pick_diverse_sample(clean_queries, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):
    """
    Mengambil sampel acak dengan seed tetap dan memastikan kategorinya
    tidak terlalu seragam (minimal MIN_UNIQUE_CATEGORIES kategori berbeda).
    """
    if len(clean_queries) < sample_size:
        raise ValueError(
            f"Kueri bersih hanya {len(clean_queries)}, kurang dari {sample_size}."
        )

    rng = random.Random(seed)
    available_categories = len({item["category"] for item in clean_queries})
    target_categories = min(MIN_UNIQUE_CATEGORIES, available_categories)

    # Coba tarik sampel deterministik sampai variasi kategori terpenuhi.
    for attempt in range(1, 100):
        selected = rng.sample(clean_queries, sample_size)
        unique_categories = {item["category"] for item in selected}
        if len(unique_categories) >= target_categories:
            return selected, attempt

    return rng.sample(clean_queries, sample_size), 100


def main():
    if not QUERY_PATH.exists():
        print(f"[ERROR] Berkas kueri tidak ditemukan: {QUERY_PATH}")
        return

    if not ARTICLE_PATH.exists():
        print(f"[ERROR] Berkas artikel tidak ditemukan: {ARTICLE_PATH}")
        return

    articles = load_articles(ARTICLE_PATH)
    clean_queries, skipped = load_clean_queries(QUERY_PATH, articles)

    selected, attempt = pick_diverse_sample(
        clean_queries,
        sample_size=SAMPLE_SIZE,
        seed=RANDOM_SEED,
    )

    categories = sorted({item["category"] for item in selected})

    print("========== SAMPEL TABEL 3.4 ==========")
    print(f"Berkas sumber          : {QUERY_PATH.name}")
    print(f"Seed acak              : {RANDOM_SEED} (percobaan ke-{attempt})")
    print(f"Kueri bersih tersedia  : {len(clean_queries)} (dilewati/audit: {skipped})")
    print(f"Variasi kategori ({len(categories)})  : {', '.join(categories)}")
    print("\n--- FORMAT BARIS (ID | Kategori | Kueri) ---")
    for item in selected:
        print(
            f"{item['source_article_id']} | "
            f"{item['category']} | "
            f"{item['query']}"
        )


if __name__ == "__main__":
    main()

