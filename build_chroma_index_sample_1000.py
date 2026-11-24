"""
Membangun indeks vektor ChromaDB untuk korpus pilot (1.000 artikel, 3.985 chunk).
Menggunakan embedding paraphrase-multilingual-MiniLM-L12-v2 yang sudah tersimpan
pada file .npy dan metadata dari file .jsonl (tanpa inferensi ulang model).
"""

import argparse
import datetime
import json
from pathlib import Path
import time
import chromadb
import numpy as np


# =========================================================
# KONFIGURASI JALUR BERKAS
# =========================================================
ROOT_DIR = Path(__file__).resolve().parent

METADATA_PATH = ROOT_DIR / "models" / "chunk_metadata_sample_1000.jsonl"
EMBEDDINGS_PATH = ROOT_DIR / "models" / "chunk_embeddings_sample_1000.npy"
CHROMA_DIR = ROOT_DIR / "chroma_db" / "sample_1000"
CONFIG_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "chroma_config_sample_1000.json"

COLLECTION_NAME = "pilot_chunks_sample_1000"
BATCH_SIZE = 1000

# Parameter HNSW eksplisit
HNSW_SPACE = "cosine"
HNSW_EF_CONSTRUCTION = 100
HNSW_M = 16
HNSW_EF_SEARCH = 100


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
                    raise ValueError(
                        f"Gagal mem-parsing baris {nomor_baris} pada {path}: {err}"
                    )
    return data


def simpan_json_rapi(data, path: Path):
    """Menyimpan data Python dictionary ke berkas JSON dengan indentasi rapi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def periksa_berkas_input():
    """Memeriksa keberadaan berkas metadata dan embedding."""
    for path in (METADATA_PATH, EMBEDDINGS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Berkas masukan tidak ditemukan: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Membangun indeks vektor ChromaDB untuk korpus pilot 1.000 artikel."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Bangun ulang koleksi jika sudah ada sebelumnya di direktori ChromaDB.",
    )
    args = parser.parse_args()

    periksa_berkas_input()

    print("==================================================")
    print("MEMBANGUN INDEKS CHROMADB PILOT (SAMPLE 1.000)")
    print("==================================================")
    print(f"Versi ChromaDB terpasang: {chromadb.__version__}")
    print(f"Direktori database      : {CHROMA_DIR}")
    print(f"Nama koleksi            : {COLLECTION_NAME}")

    # 1. Muat metadata dan embedding yang sudah ada
    print("\n[1/4] Memuat metadata dan embedding dari disk...")
    metadata_rows = baca_jsonl(METADATA_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    total_chunks = len(metadata_rows)
    if total_chunks != embeddings.shape[0]:
        raise ValueError(
            f"Jumlah baris metadata ({total_chunks}) tidak sama dengan "
            f"jumlah baris embedding ({embeddings.shape[0]})."
        )

    if embeddings.shape[1] != 384:
        raise ValueError(
            f"Dimensi embedding tidak sesuai. Diharapkan 384, ditemukan {embeddings.shape[1]}."
        )

    print(f"Total chunk terdeteksi  : {total_chunks}")
    print(f"Dimensi embedding       : {embeddings.shape[1]}")

    # 2. Inisialisasi PersistentClient
    print("\n[2/4] Menginisialisasi ChromaDB PersistentClient...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Periksa batas batch bawaan client
    max_batch = client.get_max_batch_size()
    batch_size = min(BATCH_SIZE, max_batch)
    print(f"Batas batch client      : {max_batch} (digunakan batch_size: {batch_size})")

    # Periksa apakah koleksi sudah ada
    koleksi_ada = False
    try:
        koleksi_lama = client.get_collection(COLLECTION_NAME)
        koleksi_ada = True
    except Exception:
        koleksi_ada = False

    if koleksi_ada:
        if not args.rebuild:
            print(
                f"\nPERINGATAN: Koleksi '{COLLECTION_NAME}' sudah ada di database "
                f"({koleksi_lama.count()} item).\n"
                f"Proses dihentikan agar tidak menimpa data diam-diam.\n"
                f"Gunakan argumen '--rebuild' jika ingin menghapus dan membangun ulang koleksi."
            )
            return
        else:
            print(f"Menghapus koleksi lama '{COLLECTION_NAME}' karena opsi --rebuild aktif...")
            client.delete_collection(COLLECTION_NAME)

    # 3. Buat koleksi dengan parameter HNSW eksplisit
    print("\n[3/4] Membuat koleksi baru dengan metrik Cosine HNSW...")
    metadata_koleksi = {
        "hnsw:space": HNSW_SPACE,
        "hnsw:construction_ef": HNSW_EF_CONSTRUCTION,
        "hnsw:M": HNSW_M,
        "hnsw:search_ef": HNSW_EF_SEARCH,
    }

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata=metadata_koleksi,
    )

    # 4. Memasukkan data ke dalam koleksi secara batch
    print(f"\n[4/4] Memasukkan {total_chunks} chunk ke ChromaDB...")
    waktu_mulai = time.time()

    for awal_indeks in range(0, total_chunks, batch_size):
        akhir_indeks = min(awal_indeks + batch_size, total_chunks)

        batch_ids = []
        batch_documents = []
        batch_metadatas = []
        batch_embeddings = []

        for idx in range(awal_indeks, akhir_indeks):
            meta = metadata_rows[idx]
            batch_ids.append(str(meta["chunk_id"]))
            batch_documents.append(str(meta.get("text", "")))
            batch_metadatas.append({
                "article_id": str(meta["article_id"]),
                "chunk_index": int(meta["chunk_index"]),
                "category": str(meta.get("category", "")),
                "source": str(meta.get("source", "")),
            })
            batch_embeddings.append(embeddings[idx].tolist())

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )
        print(f"  Tersimpan: {akhir_indeks}/{total_chunks} chunk...")

    durasi_detik = time.time() - waktu_mulai

    # 5. Verifikasi integritas koleksi
    jumlah_tersimpan = collection.count()
    if jumlah_tersimpan != total_chunks:
        raise ValueError(
            f"Verifikasi gagal: jumlah item di koleksi ({jumlah_tersimpan}) "
            f"tidak sama dengan total input ({total_chunks})."
        )

    # Ambil konfigurasi aktual dari koleksi
    hnsw_aktual = collection.configuration.get("hnsw", {}) if hasattr(collection, "configuration") else {}

    print("\n==================================================")
    print("VERIFIKASI & SELESAI")
    print("==================================================")
    print(f"Jumlah item tersimpan : {jumlah_tersimpan} chunk (Sesuai)")
    print(f"Durasi indexing       : {durasi_detik:.2f} detik")
    print(f"HNSW aktual           : {hnsw_aktual}")

    # 6. Simpan ringkasan konfigurasi ke berkas JSON
    catatan_konfigurasi = {
        "chromadb_version": chromadb.__version__,
        "collection_name": COLLECTION_NAME,
        "database_dir": str(CHROMA_DIR),
        "total_items": jumlah_tersimpan,
        "embedding_dimension": int(embeddings.shape[1]),
        "batch_size_used": batch_size,
        "build_duration_seconds": round(durasi_detik, 2),
        "created_at": datetime.datetime.now().isoformat(),
        "hnsw_requested_parameters": metadata_koleksi,
        "hnsw_actual_configuration": hnsw_aktual,
    }

    simpan_json_rapi(catatan_konfigurasi, CONFIG_OUTPUT_PATH)
    print(f"Konfigurasi indeks tersimpan di: {CONFIG_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

