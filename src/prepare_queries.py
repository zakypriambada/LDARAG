
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================
# KONFIGURASI
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR / "data" / "processed"
    / "articles_processed.jsonl"
)

OUTPUT_PATH = (
    BASE_DIR / "data" / "processed"
    / "queries_pilot.jsonl"
)

MODEL_NAME = "gemini-3.1-flash-lite"
SAMPLE_SIZE = 30
QUERIES_PER_ARTICLE = 2
RANDOM_SEED = 42
MAX_RETRIES = 3

load_dotenv(BASE_DIR / ".env")


# =========================
# BACA DATA
# =========================

def read_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def write_jsonl_append(path, record):
    with open(path, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


# =========================
# PILIH SAMPEL
# =========================

def select_articles(articles, sample_size=30):
    """
    Mengambil sampel dengan alokasi kategori
    yang mendekati proporsi dataset.
    """
    rng = random.Random(RANDOM_SEED)

    groups = {}
    for article in articles:
        summary = article.get("summary")
        article_id = article.get("id")

        if not article_id or not summary:
            continue

        # Summary di data processed umumnya sudah berupa teks.
        # Jika masih berupa struktur nested, ubah ke teks.
        if isinstance(summary, list):
            summary = flatten_text(summary)

        summary = str(summary).strip()
        if not summary:
            continue

        item = dict(article)
        item["summary"] = summary

        category = str(article.get("category", "unknown"))
        groups.setdefault(category, []).append(item)

    valid_articles = [
        article
        for group in groups.values()
        for article in group
    ]

    if len(valid_articles) < sample_size:
        raise ValueError(
            f"Hanya tersedia {len(valid_articles)} artikel valid."
        )

    # Pengacakan tetap sama untuk seed yang sama.
    for group in groups.values():
        rng.shuffle(group)

    # Alokasi proporsional dengan metode sisa terbesar.
    total = len(valid_articles)
    quotas = {
        category: (len(group) * sample_size / total)
        for category, group in groups.items()
    }

    allocation = {
        category: int(quota)
        for category, quota in quotas.items()
    }

    remaining = sample_size - sum(allocation.values())

    order = sorted(
        groups.keys(),
        key=lambda category: (
            quotas[category] - allocation[category]
        ),
        reverse=True,
    )

    for category in order[:remaining]:
        allocation[category] += 1

    selected = []
    for category, group in groups.items():
        selected.extend(group[:allocation[category]])

    rng.shuffle(selected)
    return selected


def flatten_text(value):
    """Menggabungkan token bersarang menjadi teks."""
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return " ".join(
            part
            for item in value
            if (part := flatten_text(item)).strip()
        )

    return ""


# =========================
# GENERASI QUERY
# =========================

PROMPT_TEMPLATE = """
Buat tepat 2 pertanyaan pencarian berita berbahasa Indonesia
berdasarkan ringkasan artikel berikut.

Kriteria:
- Terdengar alami seperti pertanyaan pengguna sungguhan.
- Kedua pertanyaan memiliki formulasi atau fokus yang berbeda.
- Pertanyaan harus dapat dijawab berdasarkan ringkasan.
- Jangan menambahkan fakta yang tidak didukung ringkasan.
- Hindari menyalin kalimat ringkasan secara langsung.
- Jangan menjawab pertanyaannya.
- Jangan membuat pertanyaan terlalu umum.
- Kembalikan hanya JSON sesuai skema yang diminta.

Ringkasan artikel:
{summary}
"""


def generate_queries(client, summary):
    prompt = PROMPT_TEMPLATE.format(summary=summary)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "queries": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    }
                },
                "required": ["queries"],
            },
            temperature=0.7,
            max_output_tokens=256,
        ),
    )

    if not response.text:
        raise ValueError("Gemini mengembalikan output kosong.")

    result = json.loads(response.text)
    queries = result.get("queries", [])

    queries = [
        q.strip()
        for q in queries
        if isinstance(q, str) and q.strip()
    ]

    if len(queries) != QUERIES_PER_ARTICLE:
        raise ValueError(
            f"Jumlah query tidak sesuai: {len(queries)}"
        )

    usage = response.usage_metadata

    token_info = {
        "input_tokens": (
            getattr(usage, "prompt_token_count", None)
            if usage else None
        ),
        "output_tokens": (
            getattr(usage, "candidates_token_count", None)
            if usage else None
        ),
        "total_tokens": (
            getattr(usage, "total_token_count", None)
            if usage else None
        ),
    }

    return queries, token_info


# =========================
# PROSES UTAMA
# =========================

def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY belum diatur di file .env"
        )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    articles = read_jsonl(INPUT_PATH)
    selected = select_articles(articles, SAMPLE_SIZE)

    # Artikel yang sudah berhasil disimpan tidak diulang.
    completed = set()

    if OUTPUT_PATH.exists():
        for row in read_jsonl(OUTPUT_PATH):
            if (
                len(row.get("queries", []))
                == QUERIES_PER_ARTICLE
            ):
                completed.add(str(row["source_article_id"]))

    client = genai.Client(api_key=api_key)

    total_input_tokens = 0
    total_output_tokens = 0
    successful = 0
    failed = 0

    print(f"Model: {MODEL_NAME}")
    print(f"Artikel terpilih: {len(selected)}")
    print(f"Artikel sudah selesai: {len(completed)}")

    for index, article in enumerate(selected, start=1):
        article_id = str(article["id"])

        if article_id in completed:
            continue

        summary = article["summary"]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                queries, tokens = generate_queries(
                    client, summary
                )

                record = {
                    "source_article_id": article_id,
                    "category": article.get("category"),
                    "queries": queries,
                    "model": MODEL_NAME,
                    "generation_method": "gemini_api",
                    "random_seed": RANDOM_SEED,
                    "token_usage": tokens,
                    "status": "generated_needs_review",
                }

                write_jsonl_append(OUTPUT_PATH, record)

                if tokens["input_tokens"] is not None:
                    total_input_tokens += tokens["input_tokens"]

                if tokens["output_tokens"] is not None:
                    total_output_tokens += tokens["output_tokens"]

                successful += 1

                print(
                    f"[{index}/{len(selected)}] "
                    f"Berhasil: {article_id} | "
                    f"token: {tokens['total_tokens']}"
                )
                break

            except Exception as error:
                print(
                    f"[{index}/{len(selected)}] "
                    f"Percobaan {attempt} gagal "
                    f"untuk artikel {article_id}: {error}"
                )

                if attempt < MAX_RETRIES:
                    time.sleep(2 * attempt)
                else:
                    failed += 1

        time.sleep(5)

    print("\n=== RINGKASAN UJI COBA ===")
    print(f"Artikel berhasil: {successful}")
    print(f"Artikel gagal: {failed}")
    print(f"Total input token: {total_input_tokens}")
    print(f"Total output token: {total_output_tokens}")
    print(f"File hasil: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
