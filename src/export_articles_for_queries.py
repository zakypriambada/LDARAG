
import json
import random
from pathlib import Path


# =====================================
# KONFIGURASI
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "articles_sample_1000.jsonl"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_PATH = (
    OUTPUT_DIR / "articles_for_query_generation.txt"
)

SAMPLE_SIZE = 30
ARTICLES_PER_GROUP = 5
RANDOM_SEED = 42


# =====================================
# FUNGSI BANTU
# =====================================

def flatten_text(value):
    """Mengubah teks atau token bersarang menjadi teks biasa."""
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts = [flatten_text(item) for item in value]
        return " ".join(part for part in parts if part.strip())

    return ""


def load_articles(path):
    """Membaca file JSONL dan mengambil artikel yang valid."""
    articles = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                article = json.loads(line)
            except json.JSONDecodeError:
                print(f"JSON tidak valid pada baris {line_number}")
                continue

            article_id = article.get("id")
            summary = flatten_text(article.get("summary", ""))
            category = article.get("category", "Tidak diketahui")

            if not article_id or not summary.strip():
                continue

            articles.append({
                "id": str(article_id),
                "category": str(category),
                "summary": summary.strip(),
            })

    # Mencegah ID artikel yang sama masuk lebih dari sekali.
    unique_articles = {}
    for article in articles:
        unique_articles.setdefault(article["id"], article)

    return list(unique_articles.values())


def select_articles(articles):
    """
    Memilih artikel secara proporsional berdasarkan kategori.
    Seed tetap membuat pemilihan dapat direproduksi.
    """
    rng = random.Random(RANDOM_SEED)

    groups = {}
    for article in articles:
        category = article["category"]
        groups.setdefault(category, []).append(article)

    total = len(articles)

    if total < SAMPLE_SIZE:
        raise ValueError(
            f"Artikel valid hanya {total}, "
            f"sedangkan dibutuhkan {SAMPLE_SIZE}."
        )

    # Acak isi setiap kategori.
    for category_articles in groups.values():
        rng.shuffle(category_articles)

    # Hitung alokasi proporsional per kategori.
    quotas = {
        category: len(items) * SAMPLE_SIZE / total
        for category, items in groups.items()
    }

    allocation = {
        category: int(quota)
        for category, quota in quotas.items()
    }

    remaining = SAMPLE_SIZE - sum(allocation.values())

    priority = sorted(
        groups,
        key=lambda category: (
            quotas[category] - allocation[category]
        ),
        reverse=True,
    )

    for category in priority[:remaining]:
        allocation[category] += 1

    selected = []

    for category, items in groups.items():
        selected.extend(items[:allocation[category]])

    rng.shuffle(selected)

    return selected


def export_articles(articles, output_path):
    """Menyimpan artikel dalam kelompok lima artikel."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        total_groups = (
            len(articles) + ARTICLES_PER_GROUP - 1
        ) // ARTICLES_PER_GROUP

        for start in range(0, len(articles), ARTICLES_PER_GROUP):
            group_number = start // ARTICLES_PER_GROUP + 1
            group = articles[
                start:start + ARTICLES_PER_GROUP
            ]

            file.write("=" * 70 + "\n")
            file.write(
                f"KELOMPOK {group_number} DARI {total_groups}\n"
            )
            file.write("=" * 70 + "\n\n")

            for index, article in enumerate(group, start=1):
                file.write(f"Artikel kelompok: {index}\n")
                file.write(f"ID: {article['id']}\n")
                file.write(f"Kategori: {article['category']}\n")
                file.write("Ringkasan:\n")
                file.write(article["summary"] + "\n")
                file.write("\n" + "-" * 70 + "\n\n")

    print(f"File berhasil dibuat:\n{output_path}")


# =====================================
# PROGRAM UTAMA
# =====================================

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File dataset tidak ditemukan:\n{INPUT_PATH}"
        )

    articles = load_articles(INPUT_PATH)

    print(f"Artikel valid yang ditemukan: {len(articles)}")

    selected = select_articles(articles)

    export_articles(selected, OUTPUT_PATH)

    print(f"Jumlah artikel terpilih: {len(selected)}")
    print(f"Jumlah kelompok: "
          f"{(len(selected) + ARTICLES_PER_GROUP - 1) // ARTICLES_PER_GROUP}")

    print("\nDistribusi kategori:")
    distribution = {}

    for article in selected:
        category = article["category"]
        distribution[category] = (
            distribution.get(category, 0) + 1
        )

    for category, count in sorted(distribution.items()):
        print(f"- {category}: {count}")


if __name__ == "__main__":
    main()
