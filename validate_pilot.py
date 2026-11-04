
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

QUERY_FILE = BASE_DIR / "data" / "processed" / "queries_pilot_fixed.jsonl"
ARTICLE_FILE = BASE_DIR / "data" / "processed" / "articles_sample_1000.jsonl"


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON tidak valid pada {path.name}, "
                    f"baris {line_number}: {error}"
                ) from error

    return rows


def main():
    queries = read_jsonl(QUERY_FILE)
    articles = read_jsonl(ARTICLE_FILE)

    article_ids = {
        str(article["id"])
        for article in articles
        if article.get("id") is not None
    }

    missing_fields = []
    missing_article_ids = []

    for number, item in enumerate(queries, start=1):
        if not item.get("source_article_id") or not item.get("query"):
            missing_fields.append(number)
            continue

        source_id = str(item["source_article_id"])

        if source_id not in article_ids:
            missing_article_ids.append(
                (number, source_id)
            )

    print(f"Jumlah artikel: {len(articles)}")
    print(f"Jumlah kueri: {len(queries)}")
    print(f"Kueri tanpa field wajib: {len(missing_fields)}")
    print(f"ID artikel sumber tidak ditemukan: {len(missing_article_ids)}")

    if missing_fields:
        print("Nomor kueri dengan field yang kurang:", missing_fields)

    if missing_article_ids:
        for number, source_id in missing_article_ids:
            print(f"Kueri {number}: ID tidak ditemukan: {source_id}")

    if (
        len(queries) == 60
        and not missing_fields
        and not missing_article_ids
    ):
        print("\nVALIDASI BERHASIL: 60 kueri siap untuk pilot test.")
    else:
        print("\nValidasi perlu diperiksa sebelum retrieval dijalankan.")


if __name__ == "__main__":
    main()
