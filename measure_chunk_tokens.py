import json
import statistics
from pathlib import Path

from sentence_transformers import SentenceTransformer

import config as cfg


# =========================================================
# KONFIGURASI
# =========================================================
CHUNKS_PATH = cfg.CHUNKS_PATH
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_SAMPLES = 2000


def main():
    if not CHUNKS_PATH.exists():
        print(f"[ERROR] Berkas chunk tidak ditemukan: {CHUNKS_PATH}")
        print(
            "Silakan buat berkas chunks_sample_10000.jsonl terlebih dahulu "
            "dari articles_sample_10000.jsonl sebelum menjalankan skrip ini."
        )
        return

    print(f"Memuat model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    batas = model.max_seq_length

    panjang = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAX_SAMPLES:
                break
            if not line.strip():
                continue
            teks = json.loads(line)["text"]
            token_ids = model.tokenizer(teks, add_special_tokens=True)["input_ids"]
            panjang.append(len(token_ids))

    if not panjang:
        print("[ERROR] Tidak ada data chunk yang berhasil dibaca.")
        return

    lebih = sum(p > batas for p in panjang)

    print("\n========== HASIL PENGUKURAN TOKEN CHUNK ==========")
    print(f"Berkas sumber       : {CHUNKS_PATH.name}")
    print(f"Jumlah sampel chunk : {len(panjang)}")
    print(
        f"batas token: {batas} | "
        f"rata-rata: {round(statistics.mean(panjang), 1)} | "
        f"median: {statistics.median(panjang)}"
    )
    print(
        f"chunk melebihi batas: {lebih}/{len(panjang)} "
        f"({lebih / len(panjang):.1%})"
    )


if __name__ == "__main__":
    main()

