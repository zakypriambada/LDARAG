"""
Membandingkan hasil baseline eksak (NumPy) dengan baseline aproksimasi (ChromaDB)
untuk korpus pilot 1.000 artikel dan 60 kueri.
Menganalisis irisan Top-20, urutan identik, selisih skor, pergeseran peringkat artikel sumber,
serta metrik Hit@K dan MRR berdampingan.
"""

import json
from pathlib import Path
import numpy as np


# =========================================================
# KONFIGURASI JALUR BERKAS
# =========================================================
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "processed"

NUMPY_RESULTS_PATH = DATA_DIR / "baseline_results_sample_1000.jsonl"
NUMPY_METRICS_PATH = DATA_DIR / "baseline_metrics_sample_1000.json"

CHROMA_VARIANTS = {
    "efdefault": {
        "results_path": DATA_DIR / "baseline_results_sample_1000_chroma_efdefault.jsonl",
        "metrics_path": DATA_DIR / "baseline_metrics_sample_1000_chroma_efdefault.json",
        "label": "ChromaDB (ef_default)",
    },
    "ef100": {
        "results_path": DATA_DIR / "baseline_results_sample_1000_chroma_ef100.jsonl",
        "metrics_path": DATA_DIR / "baseline_metrics_sample_1000_chroma_ef100.json",
        "label": "ChromaDB (ef_search=100)",
    },
}

OUTPUT_COMPARE_PATH = DATA_DIR / "compare_chroma_vs_numpy_sample_1000.json"


def baca_jsonl(path: Path):
    """Membaca berkas JSONL dan mengembalikan daftar dictionary."""
    data = []
    with path.open("r", encoding="utf-8") as file:
        for nomor_baris, baris in enumerate(file, start=1):
            baris = baris.strip()
            if baris:
                try:
                    data.append(json.loads(baris))
                except json.JSONDecodeError as err:
                    raise ValueError(f"Gagal mem-parsing baris {nomor_baris} pada {path}: {err}")
    return data


def baca_json(path: Path):
    """Membaca berkas JSON."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def simpan_json(data: dict, path: Path):
    """Menyimpan data Python dictionary ke berkas JSON dengan format rapi."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def bandingkan_varian(numpy_results, chroma_results, nama_varian, label_varian, numpy_metrics, chroma_metrics):
    """Membandingkan satu varian Chroma dengan NumPy secara detail."""
    if len(numpy_results) != len(chroma_results):
        raise ValueError(
            f"Jumlah kueri berbeda: NumPy ({len(numpy_results)}) vs Chroma ({len(chroma_results)})"
        )

    total_queries = len(numpy_results)
    overlap_ratios = []
    identical_order_count = 0
    score_diffs = []
    rank_differences = []

    for q_idx in range(total_queries):
        np_row = numpy_results[q_idx]
        ch_row = chroma_results[q_idx]

        qid = np_row["query_id"]

        np_cands = np_row["candidates"]
        ch_cands = ch_row["candidates"]

        np_ids = [c["chunk_id"] for c in np_cands]
        ch_ids = [c["chunk_id"] for c in ch_cands]

        # 1. Irisan Top-20 chunk_id
        set_np = set(np_ids)
        set_ch = set(ch_ids)
        intersection = set_np.intersection(set_ch)
        overlap_ratio = len(intersection) / float(len(set_np)) if set_np else 0.0
        overlap_ratios.append(overlap_ratio)

        # 2. Urutan Top-20 identik persis
        if np_ids == ch_ids:
            identical_order_count += 1

        # 3. Selisih skor pada chunk yang sama
        np_scores_map = {c["chunk_id"]: c["semantic_score"] for c in np_cands}
        ch_scores_map = {c["chunk_id"]: c["semantic_score"] for c in ch_cands}

        for cid in intersection:
            diff = abs(np_scores_map[cid] - ch_scores_map[cid])
            score_diffs.append(diff)

        # 4. Perbedaan source_article_rank
        np_rank = np_row["source_article_rank"]
        ch_rank = ch_row["source_article_rank"]
        if np_rank != ch_rank:
            rank_differences.append({
                "query_id": qid,
                "query": np_row["query"],
                "source_article_id": np_row["source_article_id"],
                "rank_numpy": np_rank,
                "rank_chroma": ch_rank,
            })

    hasil_komparasi = {
        "variant": nama_varian,
        "label": label_varian,
        "total_queries": total_queries,
        "mean_top20_chunk_overlap": float(np.mean(overlap_ratios)),
        "min_top20_chunk_overlap": float(np.min(overlap_ratios)),
        "max_top20_chunk_overlap": float(np.max(overlap_ratios)),
        "identical_order_queries_count": identical_order_count,
        "identical_order_queries_percentage": (identical_order_count / total_queries) * 100.0,
        "max_score_difference_matching_chunks": float(np.max(score_diffs)) if score_diffs else 0.0,
        "mean_score_difference_matching_chunks": float(np.mean(score_diffs)) if score_diffs else 0.0,
        "rank_difference_queries_count": len(rank_differences),
        "rank_difference_queries": rank_differences,
        "metrics_comparison": {
            "Hit@1": {
                "numpy": numpy_metrics.get("Hit@1"),
                "chroma": chroma_metrics.get("Hit@1"),
                "diff": chroma_metrics.get("Hit@1", 0) - numpy_metrics.get("Hit@1", 0),
            },
            "Hit@3": {
                "numpy": numpy_metrics.get("Hit@3"),
                "chroma": chroma_metrics.get("Hit@3"),
                "diff": chroma_metrics.get("Hit@3", 0) - numpy_metrics.get("Hit@3", 0),
            },
            "Hit@5": {
                "numpy": numpy_metrics.get("Hit@5"),
                "chroma": chroma_metrics.get("Hit@5"),
                "diff": chroma_metrics.get("Hit@5", 0) - numpy_metrics.get("Hit@5", 0),
            },
            "Hit@20": {
                "numpy": numpy_metrics.get("Hit@20"),
                "chroma": chroma_metrics.get("Hit@20"),
                "diff": chroma_metrics.get("Hit@20", 0) - numpy_metrics.get("Hit@20", 0),
            },
            "MRR": {
                "numpy": numpy_metrics.get("MRR"),
                "chroma": chroma_metrics.get("MRR"),
                "diff": chroma_metrics.get("MRR", 0) - numpy_metrics.get("MRR", 0),
            },
        },
    }

    return hasil_komparasi


def main():
    print("==================================================")
    print("KOMPARASI BASELINE: NUMPY (EKSAK) VS CHROMADB")
    print("==================================================")

    if not NUMPY_RESULTS_PATH.exists() or not NUMPY_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Berkas baseline NumPy tidak ditemukan: {NUMPY_RESULTS_PATH} atau {NUMPY_METRICS_PATH}"
        )

    numpy_results = baca_jsonl(NUMPY_RESULTS_PATH)
    numpy_metrics = baca_json(NUMPY_METRICS_PATH)
    print(f"NumPy baseline dimuat: {len(numpy_results)} kueri")

    semua_hasil = {}

    for var_key, var_cfg in CHROMA_VARIANTS.items():
        if not var_cfg["results_path"].exists() or not var_cfg["metrics_path"].exists():
            raise FileNotFoundError(f"Berkas Chroma tidak ditemukan: {var_cfg['results_path']}")

        chroma_results = baca_jsonl(var_cfg["results_path"])
        chroma_metrics = baca_json(var_cfg["metrics_path"])

        hasil = bandingkan_varian(
            numpy_results=numpy_results,
            chroma_results=chroma_results,
            nama_varian=var_key,
            label_varian=var_cfg["label"],
            numpy_metrics=numpy_metrics,
            chroma_metrics=chroma_metrics,
        )
        semua_hasil[var_key] = hasil

        print(f"\n--------------------------------------------------")
        print(f"HASIL KOMPARASI: NUMPY vs {var_cfg['label']}")
        print(f"--------------------------------------------------")
        print(f"Rata-rata Irisan Top-20 Chunk : {hasil['mean_top20_chunk_overlap'] * 100:.2f}%")
        print(f"Urutan Top-20 Identik Persis  : {hasil['identical_order_queries_count']}/{hasil['total_queries']} kueri ({hasil['identical_order_queries_percentage']:.1f}%)")
        print(f"Selisih Maksimum Skor Chunk   : {hasil['max_score_difference_matching_chunks']:.2e}")
        print(f"Selisih Rata-rata Skor Chunk  : {hasil['mean_score_difference_matching_chunks']:.2e}")
        print(f"Jumlah Kueri Beda Rank Sumber : {hasil['rank_difference_queries_count']}")

        if hasil["rank_difference_queries_count"] > 0:
            print("\nDaftar Kueri dengan Perbedaan Peringkat Artikel:")
            for item in hasil["rank_difference_queries"]:
                print(f"  - Kueri #{item['query_id']}: Rank NumPy={item['rank_numpy']} -> Rank Chroma={item['rank_chroma']}")
        else:
            print("Peringkat artikel sumber: 100% SAMA pada semua 60 kueri.")

    # Simpan hasil perbandingan ke JSON
    simpan_json(semua_hasil, OUTPUT_COMPARE_PATH)
    print(f"\nHasil perbandingan lengkap tersimpan di: {OUTPUT_COMPARE_PATH}")

    # Cetak tabel perbandingan metrik berdampingan
    print("\n=========================================================================")
    print("TABEL PERBANDINGAN METRIK RETRIEVAL BERDAMPINGAN")
    print("=========================================================================")
    header = f"{'Metrik':<10} | {'NumPy (Eksak)':<15} | {'Chroma (efdefault)':<20} | {'Chroma (ef100)':<15}"
    print(header)
    print("-" * len(header))

    metrics_keys = ["Hit@1", "Hit@3", "Hit@5", "Hit@20", "MRR"]
    for m in metrics_keys:
        val_np = numpy_metrics.get(m, 0.0)
        val_ch_def = semua_hasil["efdefault"]["metrics_comparison"][m]["chroma"]
        val_ch_100 = semua_hasil["ef100"]["metrics_comparison"][m]["chroma"]
        print(f"{m:<10} | {val_np:<15.4f} | {val_ch_def:<20.4f} | {val_ch_100:<15.4f}")

    print("=========================================================================")


if __name__ == "__main__":
    main()

