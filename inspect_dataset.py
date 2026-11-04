
import json
from pathlib import Path

# Ganti dengan lokasi file train01.jsonl di laptop kamu
DATA_PATH = Path("data/raw/train.01.jsonl")


def flatten_tokens(data):
    """Menggabungkan token kata bertingkat menjadi teks biasa."""
    tokens = []

    def collect(item):
        if isinstance(item, list):
            for child in item:
                collect(child)
        elif isinstance(item, str):
            tokens.append(item)

    collect(data)
    return " ".join(tokens)


def inspect_dataset(path):
    total = 0
    failed = 0
    required_fields = {"id", "paragraphs", "summary"}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                article = json.loads(line)
                total += 1

                if total == 1:
                    print("Field yang tersedia:")
                    print(list(article.keys()))

                    print("\nContoh ID:")
                    print(article.get("id"))

                    print("\nContoh isi artikel:")
                    print(flatten_tokens(
                        article.get("paragraphs", [])
                    )[:1000])

                    print("\nContoh ringkasan:")
                    print(flatten_tokens(
                        article.get("summary", [])
                    ))

                    print("\nField wajib tersedia:")
                    print(required_fields.issubset(article.keys()))

                if not required_fields.issubset(article.keys()):
                    failed += 1

            except json.JSONDecodeError as error:
                failed += 1
                print(
                    f"JSON tidak valid pada baris {line_number}: "
                    f"{error}"
                )

    print("\n===== HASIL PEMERIKSAAN =====")
    print(f"Jumlah record valid JSON: {total}")
    print(f"Jumlah record bermasalah: {failed}")


if __name__ == "__main__":
    if not DATA_PATH.exists():
        print(f"File tidak ditemukan: {DATA_PATH}")
        print("Periksa kembali lokasi train01.jsonl.")
    else:
        inspect_dataset(DATA_PATH)
