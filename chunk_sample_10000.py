import json
from pathlib import Path

from src.chunking import create_chunks


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "articles_sample_10000.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "chunks_sample_10000.jsonl"
)

CHUNK_SIZE = 100
OVERLAP = 20


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Berkas input tidak ditemukan: {INPUT_PATH}")

    total_articles = 0
    total_chunks = 0
    article_ids = set()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(INPUT_PATH, "r", encoding="utf-8") as infile,
        open(OUTPUT_PATH, "w", encoding="utf-8") as outfile,
    ):
        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue

            article = json.loads(line)
            article_id = article.get("id")
            text = article.get("text", "")

            if not article_id or not text.strip():
                continue

            if article_id in article_ids:
                raise ValueError(
                    f"ID artikel duplikat ditemukan pada baris {line_number}: {article_id}"
                )

            article_ids.add(article_id)
            chunks = create_chunks(
                text,
                chunk_size=CHUNK_SIZE,
                overlap=OVERLAP,
            )

            for chunk_index, chunk_text in enumerate(chunks):
                chunk = {
                    "chunk_id": f"{article_id}__chunk_{chunk_index}",
                    "article_id": article_id,
                    "category": article.get("category"),
                    "source": article.get("source"),
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                }

                outfile.write(
                    json.dumps(chunk, ensure_ascii=False) + "\n"
                )
                total_chunks += 1

            total_articles += 1

    print(f"Artikel diproses : {total_articles}")
    print(f"Total chunk      : {total_chunks}")
    print(f"Ukuran chunk     : {CHUNK_SIZE} kata")
    print(f"Overlap          : {OVERLAP} kata")
    print(f"Hasil disimpan   : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

