import collections
import json
from pathlib import Path
import re

# =========================================================
# KONFIGURASI JALUR
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
QUERY_PATH = BASE_DIR / "data" / "processed" / "queries_final.jsonl"
ARTICLE_PATH = BASE_DIR / "data" / "processed" / "articles_sample_10000.jsonl"

PENUNJUK = re.compile(
    r"\b(tersebut|ini|itu|kemarin|besok|mendatang|baru-baru ini|pekan lalu|minggu lalu|tahun lalu|akhir pekan)\b",
    re.IGNORECASE,
)


def kata(teks):
    return re.findall(r"\w+", teks.lower())


def token_kueri(q):
    """Kata berhuruf kapital (nama, judul) diberi penanda unik agar tidak dihitung sebagai salinan."""
    return [f"<NAMA{i}>" if w[0].isupper() else w.lower()
            for i, w in enumerate(re.findall(r"\w+", q))]


def runtut_terpanjang(a, b):
    """Panjang urutan kata berurutan terpanjang yang sama antara dua daftar kata."""
    terbaik = 0
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            terbaik = max(terbaik, k)
    return terbaik


def main():
    if not QUERY_PATH.exists():
        print(f"[ERROR] Berkas kueri belum ada: {QUERY_PATH}")
        return

    if not ARTICLE_PATH.exists():
        print(f"[ERROR] Berkas artikel belum ada: {ARTICLE_PATH}")
        return

    print("Memuat ringkasan artikel...")
    ringkasan = {}
    with open(ARTICLE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                ringkasan[str(item["id"])] = str(item.get("summary", ""))

    per_artikel = collections.Counter()
    bertanda, semua = [], []
    ada_penunjuk, salin_panjang, bukan_tanya = [], [], []

    with open(QUERY_PATH, "r", encoding="utf-8") as f:
        for no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            r = json.loads(line)
            sid, q = str(r["source_article_id"]), r.get("query")

            if q is None:
                bertanda.append(sid)
                continue

            per_artikel[sid] += 1
            semua.append(q)

            if sid not in ringkasan:
                print(f"[PERINGATAN] Baris {no}: ID tidak ada di korpus -> {sid}")

            if PENUNJUK.search(q):
                ada_penunjuk.append((no, q))

            if sid in ringkasan:
                n = runtut_terpanjang(token_kueri(q), kata(ringkasan[sid]))
                if n >= 6:
                    salin_panjang.append((no, n, q))

            if not q.strip().endswith("?"):
                bukan_tanya.append((no, q))

    # Menghitung jumlah kueri unik yang ditandai
    kueri_bermasalah = set()

    kueri_bermasalah.update(no for no, q in bukan_tanya)
    kueri_bermasalah.update(no for no, n, q in salin_panjang)
    kueri_bermasalah.update(no for no, q in ada_penunjuk)

    jumlah_ditandai = len(kueri_bermasalah)

    persentase = (
        jumlah_ditandai / len(semua) * 100
        if semua else 0
    )

    print("\n========== RINGKASAN AUDIT ==========")
    print(f"Total kueri              : {len(semua)}")
    print(f"Kueri ditandai unik      : {jumlah_ditandai}")
    print(f"Persentase ditandai      : {persentase:.2f}%")

    print("\n========== HASIL AUDIT KUERI ==========")
    print(f"Total kueri terkumpul       : {len(semua)}")
    print(f"Jumlah artikel unik         : {len(per_artikel)}")
    print(f"Artikel bukan tepat 2 kueri : {[s for s, c in per_artikel.items() if c != 2]}")
    print(f"Duplikat kueri              : {[q for q, c in collections.Counter(semua).items() if c > 1]}")
    print(f"Artikel ditandai kosong     : {bertanda}")

    print(f"\n[1] Tidak diakhiri tanda tanya ({len(bukan_tanya)}):")
    for no, q in bukan_tanya:
        print(f"    - baris {no:3d}: {q}")

    print(f"\n[2] Menyalin >= 6 kata berurutan dari ringkasan ({len(salin_panjang)}):")
    for no, n, q in salin_panjang:
        print(f"    - baris {no:3d} ({n} kata): {q}")

    print(f"\n[3] Mengandung kata penunjuk / waktu relatif ({len(ada_penunjuk)}):")
    for no, q in ada_penunjuk:
        print(f"    - baris {no:3d}: {q}")


if __name__ == "__main__":
    main()