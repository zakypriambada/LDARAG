
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "data" / "processed" / "queries_pilot.json"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "queries_pilot_fixed.jsonl"

def main():
    with open(FILE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        raise ValueError(
            'Format harus berupa objek JSON dengan array "queries".'
        )

    queries = data["queries"]

    for index, item in enumerate(queries, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Kueri ke-{index} bukan objek JSON.")
        if not item.get("source_article_id") or not item.get("query"):
            raise ValueError(
                f"Kueri ke-{index} tidak memiliki ID artikel atau query."
            )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        for item in queries:
            file.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    print("Konversi berhasil!")
    print(f"Jumlah kueri: {len(queries)}")
    print(f"File hasil: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
