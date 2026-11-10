import json
import random
from pathlib import Path

# =========================================================
# KONFIGURASI JALUR BERKAS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

# Berkas artikel dan kueri (sesuaikan jika nama berkas kueri Anda berbeda)
ARTICLE_PATH = BASE_DIR / "data" / "processed" / "articles_sample_10000.jsonl"
QUERY_PATH = BASE_DIR / "data" / "processed" / "queries_final.jsonl"


def main():
    # 1. Pastikan kedua berkas ada sebelum diproses
    if not ARTICLE_PATH.exists():
        print(f"[ERROR] Berkas artikel tidak ditemukan: {ARTICLE_PATH}")
        return

    if not QUERY_PATH.exists():
        print(f"[ERROR] Berkas kueri belum ditemukan: {QUERY_PATH}")
        print("Pastikan Anda sudah meletakkan berkas kueri di folder data/processed/.")
        return

    # 2. Baca pemetaan ID Artikel -> Kategori
    print("Membaca data artikel...")
    kategori = {}
    with open(ARTICLE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                kategori[str(item["id"])] = item.get("category", "tidak diketahui")

    # 3. Baca seluruh kueri
    with open(QUERY_PATH, "r", encoding="utf-8") as f:
        baris = [json.loads(line) for line in f if line.strip()]

    print(f"Total kueri ditemukan : {len(baris)}")
    print(f"Total artikel terdata : {len(kategori)}")
    print("\n--- SAMPEL 5 KUERI ACAK ---")

    # 4. Ambil 5 kueri acak dan tampilkan
    random.seed(42)
    sample_queries = random.sample(baris, min(5, len(baris)))

    for r in sample_queries:
        sid = str(r.get("source_article_id"))
        nama_kategori = kategori.get(sid, "? (ID TIDAK DITEMUKAN)")
        query_text = r.get("query")
        print(f"{sid} | {nama_kategori} | {query_text}")


if __name__ == "__main__":
    main()