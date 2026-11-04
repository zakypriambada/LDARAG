
import json
from pathlib import Path
from collections import Counter

DATA_PATH = Path("data/raw/train.01.jsonl")


def flatten_tokens(data):
    tokens = []

    def collect(item):
        if isinstance(item, list):
            for child in item:
                collect(child)
        elif isinstance(item, str):
            tokens.append(item)

    collect(data)
    return " ".join(tokens)


def main():
    ids = []
    categories = Counter()
    empty_articles = 0
    short_articles = 0
    empty_summaries = 0
    article_texts = Counter()
    summary_texts = Counter()
    total = 0

    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            article = json.loads(line)
            total += 1

            article_id = article.get("id")
            if article_id:
                ids.append(article_id)

            categories[article.get("category", "Tidak diketahui")] += 1

            text = flatten_tokens(article.get("paragraphs", [])).strip()
            summary = flatten_tokens(article.get("summary", [])).strip()

            if not text:
                empty_articles += 1
            elif len(text.split()) < 20:
                short_articles += 1

            if not summary:
                empty_summaries += 1

            if text:
                article_texts[text] += 1
            if summary:
                summary_texts[summary] += 1

    duplicate_ids = len(ids) - len(set(ids))
    duplicate_articles = sum(
        count - 1 for count in article_texts.values() if count > 1
    )
    duplicate_summaries = sum(
        count - 1 for count in summary_texts.values() if count > 1
    )

    print("===== HASIL VALIDASI DATASET =====")
    print(f"Total artikel: {total}")
    print(f"ID duplikat: {duplicate_ids}")
    print(f"Artikel kosong: {empty_articles}")
    print(f"Artikel kurang dari 20 kata: {short_articles}")
    print(f"Ringkasan kosong: {empty_summaries}")
    print(f"Teks artikel duplikat: {duplicate_articles}")
    print(f"Teks ringkasan duplikat: {duplicate_summaries}")

    print("\n===== DISTRIBUSI KATEGORI =====")
    for category, count in categories.most_common():
        print(f"{category}: {count}")


if __name__ == "__main__":
    main()
