import json
import random
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


# =========================================================
# KONFIGURASI
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

RAW_PATH = BASE_DIR / "data" / "raw" / "train.01.jsonl"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "articles_processed.jsonl"
PILOT_QUERY_PATH = BASE_DIR / "data" / "processed" / "queries_pilot_fixed.jsonl"

OUTPUT_PATH = BASE_DIR / "data" / "processed" / "articles_sample_10000.jsonl"
INFO_PATH = BASE_DIR / "data" / "processed" / "sample_10000_info.json"

SAMPLE_SIZE = 10_000
SEED = 42

# Field mentah yang bukan metadata (isi artikel).
CONTENT_FIELDS = {"paragraphs", "summary", "gold_labels"}

WIB = timezone(timedelta(hours=7))
MAX_DAY_DIFF = 3  # toleransi selisih ID vs tanggal di URL (hari)

# Pola tanggal yang jelas di URL (bukan sekadar angka "20xx").
URL_DATE_PATTERNS = [
    re.compile(r"/(20\d{2})(\d{2})(\d{2})\d{6}-"),   # cnnindonesia
    re.compile(r"/(20\d{2})/(\d{2})/(\d{2})/"),      # poskotanews, suara.com
    re.compile(r"/read/(20\d{2})(\d{2})(\d{2})/"),   # rimanews
]


# =========================================================
# PEMBACAAN DATA
# =========================================================
def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_raw_metadata(path):
    """Metadata mentah per ID (semua field selain isi artikel)."""
    metadata = {}
    field_names = set()

    for row in read_jsonl(path):
        article_id = row.get("id")
        if not article_id:
            continue

        meta = {
            key: value
            for key, value in row.items()
            if key not in CONTENT_FIELDS
        }
        field_names.update(meta.keys())
        metadata[str(article_id)] = meta

    return metadata, sorted(field_names)


# =========================================================
# EKSTRAKSI TANGGAL
# =========================================================
def date_from_id(article_id):
    """Awalan ID 10 digit = unix timestamp (dikonversi ke WIB)."""
    match = re.match(r"^(\d{10})(?!\d)", str(article_id))
    if not match:
        return None

    timestamp = int(match.group(1))
    if not 946_684_800 <= timestamp <= 1_893_456_000:  # 2000-2030
        return None

    return datetime.fromtimestamp(timestamp, tz=WIB)


def date_from_url(url):
    """Tanggal dari pola eksplisit di URL; None jika tidak ada."""
    for pattern in URL_DATE_PATTERNS:
        match = pattern.search(str(url or ""))
        if match:
            try:
                return datetime(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                ).date()
            except ValueError:
                return None
    return None


# =========================================================
# PROGRAM UTAMA
# =========================================================
def main():
    for path in (RAW_PATH, PROCESSED_PATH):
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

    print("Membaca artikel terproses...")
    articles = [
        row for row in read_jsonl(PROCESSED_PATH)
        if row.get("id") and str(row.get("text", "")).strip()
    ]
    print(f"Artikel valid: {len(articles)}")

    ids = [str(a["id"]) for a in articles]
    if len(ids) != len(set(ids)):
        raise ValueError("Ada ID artikel duplikat.")

    if len(articles) < SAMPLE_SIZE:
        raise ValueError(
            f"Artikel hanya {len(articles)}, diminta {SAMPLE_SIZE}."
        )

    print("Membaca metadata mentah...")
    raw_metadata, field_names = load_raw_metadata(RAW_PATH)
    print(f"Field pada data mentah: {field_names}")

    # Pengambilan sampel acak, reproducible (sama seperti sebelumnya).
    rng = random.Random(SEED)
    selected = rng.sample(articles, SAMPLE_SIZE)

    records = []
    dates = []
    with_url_date = 0
    agree = 0
    disagree_examples = []
    disagree = 0

    for article in selected:
        article_id = str(article["id"])
        meta = raw_metadata.get(article_id, {})

        id_dt = date_from_id(article_id)
        url_date = date_from_url(meta.get("source_url"))

        record = dict(article)

        if id_dt:
            record["date"] = id_dt.date().isoformat()
            record["year"] = id_dt.year
            record["date_method"] = "id_unix_timestamp_wib"
            dates.append(id_dt.date())
        else:
            record["date"] = None
            record["year"] = None
            record["date_method"] = None

        # Validasi silang dengan tanggal eksplisit di URL.
        if id_dt and url_date:
            with_url_date += 1
            diff = abs((id_dt.date() - url_date).days)
            if diff <= MAX_DAY_DIFF:
                agree += 1
            else:
                disagree += 1
                if len(disagree_examples) < 10:
                    disagree_examples.append({
                        "id": article_id,
                        "id_date": id_dt.date().isoformat(),
                        "url_date": url_date.isoformat(),
                        "url": meta.get("source_url"),
                    })

        records.append(record)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    categories = Counter(r.get("category") for r in records)
    year_counts = Counter(d.year for d in dates)
    month_counts = Counter(d.strftime("%Y-%m") for d in dates)

    # Irisan dengan artikel sumber query pilot.
    pilot_overlap = None
    if PILOT_QUERY_PATH.exists():
        pilot_ids = {
            str(row["source_article_id"])
            for row in read_jsonl(PILOT_QUERY_PATH)
            if row.get("source_article_id")
        }
        sample_ids = {str(r["id"]) for r in records}
        pilot_overlap = {
            "pilot_source_articles": len(pilot_ids),
            "present_in_sample_10000": len(pilot_ids & sample_ids),
        }

    info = {
        "source_file": "train.01.jsonl",
        "source_articles_valid": len(articles),
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "method": "random.Random(seed).sample",
        "raw_fields": field_names,
        "date_method": "id_unix_timestamp_wib",
        "articles_with_date": len(dates),
        "date_min": min(dates).isoformat() if dates else None,
        "date_max": max(dates).isoformat() if dates else None,
        "articles_per_year": dict(sorted(year_counts.items())),
        "articles_per_month": dict(sorted(month_counts.items())),
        "articles_per_category": dict(categories),
        "url_cross_check": {
            "articles_with_explicit_url_date": with_url_date,
            "agree_within_days": MAX_DAY_DIFF,
            "agree": agree,
            "disagree": disagree,
            "disagree_examples": disagree_examples,
        },
        "pilot_overlap": pilot_overlap,
    }

    with INFO_PATH.open("w", encoding="utf-8") as file:
        json.dump(info, file, ensure_ascii=False, indent=2)

    print("\n===== HASIL SAMPEL 10.000 =====")
    print(f"Sumber valid     : {len(articles)}")
    print(f"Sampel           : {len(records)} (seed {SEED})")
    print(f"Artikel bertanggal: {len(dates)}")
    if dates:
        print(f"Tanggal awal     : {min(dates)}")
        print(f"Tanggal akhir    : {max(dates)}")
        print(f"Per tahun        : {dict(sorted(year_counts.items()))}")
    print(f"Per kategori     : {dict(categories)}")
    print(
        f"Validasi URL     : {with_url_date} artikel punya tanggal "
        f"eksplisit di URL; cocok {agree}, tidak cocok {disagree} "
        f"(toleransi {MAX_DAY_DIFF} hari)"
    )
    for example in disagree_examples:
        print(f"  tidak cocok: {example}")
    if pilot_overlap:
        print(f"Irisan pilot     : {pilot_overlap}")
    print(f"\nSampel   : {OUTPUT_PATH}")
    print(f"Ringkasan: {INFO_PATH}")


if __name__ == "__main__":
    main()