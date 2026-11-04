
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "articles_processed.jsonl"
)

SAMPLE_SIZE = 1000
RANDOM_SEED = 42


def load_articles(file_path=PROCESSED_DATA_PATH):
    """Membaca artikel dari file JSONL."""
    articles = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                article = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON tidak valid pada baris {line_number}: {exc}"
                ) from exc

            if not article.get("id") or not article.get("text"):
                continue

            articles.append(article)

    return articles


def sample_articles(articles, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):
    """Memilih sampel artikel secara acak dan reproducible."""
    if sample_size <= 0:
        raise ValueError("sample_size harus lebih dari 0")

    if len(articles) < sample_size:
        raise ValueError(
            f"Artikel tersedia hanya {len(articles)}, "
            f"sedangkan sampel diminta {sample_size}."
        )

    rng = random.Random(seed)
    return rng.sample(articles, sample_size)


def save_articles(articles, output_path):
    """Menyimpan artikel ke file JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for article in articles:
            file.write(json.dumps(article, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    articles = load_articles()
    selected_articles = sample_articles(articles)

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "articles_sample_1000.jsonl"
    )

    save_articles(selected_articles, output_path)

    print(f"Jumlah artikel tersedia : {len(articles)}")
    print(f"Jumlah artikel terpilih : {len(selected_articles)}")
    print(f"Seed                    : {RANDOM_SEED}")
    print(f"Hasil disimpan          : {output_path}")
