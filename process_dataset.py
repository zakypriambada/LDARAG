
import json
from pathlib import Path

from src.preprocessing import process_article
from src.chunking import create_chunks


# Lokasi file input dan output
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "raw" / "train.01.jsonl"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

ARTICLES_OUTPUT = OUTPUT_DIR / "articles_processed.jsonl"
CHUNKS_OUTPUT = OUTPUT_DIR / "chunks.jsonl"

CHUNK_SIZE = 100
CHUNK_OVERLAP = 20


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_articles = 0
    total_chunks = 0
    empty_articles = 0

    with (
        INPUT_PATH.open("r", encoding="utf-8") as source,
        ARTICLES_OUTPUT.open("w", encoding="utf-8") as article_file,
        CHUNKS_OUTPUT.open("w", encoding="utf-8") as chunk_file,
    ):
        for line in source:
            if not line.strip():
                continue

            raw_article = json.loads(line)
            article = process_article(raw_article)

            total_articles += 1

            # Simpan artikel yang sudah dibersihkan.
            # Ringkasan disimpan terpisah dari teks chunk retrieval.
            article_file.write(
                json.dumps(article, ensure_ascii=False) + "\n"
            )

            if not article["text"].strip():
                empty_articles += 1
                continue

            chunks = create_chunks(
                article["text"],
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )

            for index, chunk_text in enumerate(chunks):
                chunk_record = {
                    "chunk_id": f'{article["id"]}_chunk_{index}',
                    "article_id": article["id"],
                    "category": article["category"],
                    "source": article["source"],
                    "chunk_index": index,
                    "text": chunk_text,
                }

                chunk_file.write(
                    json.dumps(
                        chunk_record,
                        ensure_ascii=False,
                    ) + "\n"
                )

                total_chunks += 1

            if total_articles % 1000 == 0:
                print(
                    f"Sudah diproses: {total_articles} artikel"
                )

    print("\n===== HASIL PEMROSESAN =====")
    print(f"Artikel diproses : {total_articles}")
    print(f"Artikel kosong   : {empty_articles}")
    print(f"Total chunk      : {total_chunks}")
    print(f"Ukuran chunk     : {CHUNK_SIZE} kata")
    print(f"Overlap          : {CHUNK_OVERLAP} kata")
    print(f"\nArtikel tersimpan: {ARTICLES_OUTPUT}")
    print(f"Chunk tersimpan  : {CHUNKS_OUTPUT}")


if __name__ == "__main__":
    main()
