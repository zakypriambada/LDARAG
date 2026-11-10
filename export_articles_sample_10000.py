import json
import random
from pathlib import Path


# =========================================================
# KONFIGURASI
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
    BASE_DIR / "data" / "processed" / "articles_sample_10000.jsonl"
)
OUTPUT_PATH = (
    BASE_DIR / "data" / "processed" / "articles_for_query_generation_sample_10000.txt"
)

SAMPLE_SIZE = 150
ARTICLES_PER_GROUP = 5
RANDOM_SEED = 42


# =========================================================
# FUNGSI PEMBANTU
# =========================================================
def load_articles(path):
    """Membaca artikel dari berkas JSONL."""
    articles = []
    seen_ids = set()

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[PERINGATAN] JSON tidak valid baris {line_number}: {exc}")
                continue

            article_id = str(row.get("id", "")).strip()
            summary = str(row.get("summary", "")).strip()
            category = str(row.get("category", "tidak diketahui")).strip()

            if not article_id or not summary:
                continue

            if article_id in seen_ids:
                continue

            seen_ids.add(article_id)
            articles.append({
                "id": article_id,
                "category": category,
                "summary": summary,
            })

    return articles


def select_stratified_articles(articles, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):
    """
    Memilih artikel secara proporsional per kategori (Stratified Random Sampling).
    Menggunakan Largest Remainder Method untuk pembulatan kuota.
    """
    rng = random.Random(seed)

    # 1. Kelompokkan artikel berdasarkan kategori
    groups = {}
    for article in articles:
        groups.setdefault(article["category"], []).append(article)

    total_population = len(articles)
    if total_population < sample_size:
        raise ValueError(
            f"Jumlah artikel ({total_population}) kurang dari target sampel ({sample_size})."
        )

    # 2. Acak urutan di dalam tiap kategori (dengan seed tetap)
    for category_items in groups.values():
        rng.shuffle(category_items)

    # 3. Hitung kuota proporsional
    quotas = {
        cat: (len(items) * sample_size) / total_population
        for cat, items in groups.items()
    }

    allocation = {
        cat: int(quota)
        for cat, quota in quotas.items()
    }

    # 4. Sisa kuota dibagikan ke kategori dengan desimal tertinggi (Hamilton method)
    remaining = sample_size - sum(allocation.values())
    priority = sorted(
        groups.keys(),
        key=lambda c: (quotas[c] - allocation[c]),
        reverse=True,
    )

    for cat in priority[:remaining]:
        allocation[cat] += 1

    # 5. Ambil artikel sesuai kuota per kategori
    selected = []
    print("\n--- DISTRIBUSI ALOKASI STRATIFIKASI ---")
    for cat, count in sorted(allocation.items(), key=lambda x: -x[1]):
        pop_count = len(groups[cat])
        pct = (pop_count / total_population) * 100
        print(f"- {cat:15s}: {count:3d} artikel (populasi: {pop_count:5d} / {pct:5.2f}%)")
        selected.extend(groups[cat][:count])

    # 6. Acak susunan akhir agar kategori tersebar merata di tiap kelompok
    rng.shuffle(selected)
    return selected


def export_articles_grouped(articles, output_path, per_group=ARTICLES_PER_GROUP):
    """Mengekspor artikel ke file teks dalam format per kelompok (5 artikel)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_groups = (len(articles) + per_group - 1) // per_group

    with open(output_path, "w", encoding="utf-8") as file:
        for start in range(0, len(articles), per_group):
            group_number = (start // per_group) + 1
            group = articles[start : start + per_group]

            file.write("=" * 70 + "\n")
            file.write(f"KELOMPOK {group_number} DARI {total_groups}\n")
            file.write("=" * 70 + "\n\n")

            for idx, art in enumerate(group, start=1):
                file.write(f"Artikel kelompok: {idx}\n")
                file.write(f"ID: {art['id']}\n")
                file.write(f"Kategori: {art['category']}\n")
                file.write("Ringkasan:\n")
                file.write(art["summary"] + "\n")
                file.write("\n" + "-" * 70 + "\n\n")

    print(f"\n[SUKSES] Berkas berhasil disimpan:")
    print(f"Path: {output_path}")
    print(f"Total artikel : {len(articles)}")
    print(f"Total kelompok: {total_groups} (masing-masing {per_group} artikel)")


def main():
    print(f"Membaca berkas: {INPUT_PATH.name}")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Berkas tidak ditemukan: {INPUT_PATH}")

    articles = load_articles(INPUT_PATH)
    print(f"Total artikel valid dimuat: {len(articles)}")

    selected = select_stratified_articles(articles, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED)
    export_articles_grouped(selected, OUTPUT_PATH, per_group=ARTICLES_PER_GROUP)


if __name__ == "__main__":
    main()

