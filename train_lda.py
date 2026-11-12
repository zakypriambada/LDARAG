import hashlib
import json
import random
import re
import time

import gensim
import nltk
import numpy as np
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import config as cfg
from penanda_portal import hapus_penanda_portal


# =========================================================
# KONFIGURASI EKSPERIMEN
# =========================================================
SEED = 42

# Kandidat jumlah topik. Perluas/kurangi sesuai kurva C_v.
TOPIC_COUNTS = [5, 10, 15, 20, 25, 30, 40, 50]

PASSES = 5
ITERATIONS = 100
CHUNKSIZE = 1000
NO_BELOW = 5
NO_ABOVE = 0.5
KEEP_N = 100_000

# Isi dengan K pilihan akhir SETELAH memeriksa interpretabilitas topik
# (lihat file lda_topwords_*.txt). None = pakai C_v tertinggi.
SELECTED_K_OVERRIDE = None

# Naikkan nilai ini jika logika preprocessing diubah, agar cache
# token dan model lama tidak dipakai ulang.
PREPROCESS_VERSION = "v2-portal-regex-lower-letters-stopwords-sastrawi"

EXTRA_STOPWORDS = {
    "cnn", "indonesia", "jakarta", "sumber",
    "foto", "video", "halaman", "berita",
}

random.seed(SEED)
np.random.seed(SEED)


# =========================================================
# UTILITAS
# =========================================================
def sha256_file(path, block_size=1 << 20):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_obj(obj):
    data = json.dumps(
        obj, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def save_json_atomic(data, path):
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temp_path.replace(path)


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# DATA DAN PREPROCESSING
# =========================================================
def load_chunks():
    chunks = []

    with cfg.CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def load_indonesian_stopwords():
    try:
        words = set(stopwords.words("indonesian"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = set(stopwords.words("indonesian"))

    words.update(EXTRA_STOPWORDS)
    return words


def preprocess_text(text, stop_words, stemmer):
    text = hapus_penanda_portal(str(text))
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        token for token in text.split()
        if token not in stop_words and len(token) > 1
    ]

    return [stemmer.stem(token) for token in tokens]


def load_or_build_tokens(chunks, stop_words, chunks_hash, stopwords_hash):
    """
    Token hasil preprocessing disimpan agar stemming (yang lambat)
    tidak diulang saat proses dilanjutkan. Cache dipakai hanya bila
    isi file chunk, daftar stopword, dan versi preprocessing sama.
    """
    meta = {
        "chunks_sha256": chunks_hash,
        "stopwords_sha256": stopwords_hash,
        "preprocess_version": PREPROCESS_VERSION,
        "num_chunks": len(chunks),
    }

    if cfg.LDA_TOKENS_PATH.exists():
        with cfg.LDA_TOKENS_PATH.open("r", encoding="utf-8") as file:
            try:
                header = json.loads(file.readline())
            except json.JSONDecodeError:
                header = {}

            if header.get("meta") == meta:
                all_tokens = [
                    json.loads(line) for line in file if line.strip()
                ]
                if len(all_tokens) == len(chunks):
                    print("Cache token ditemukan; stemming dilewati.")
                    return all_tokens

        print("Cache token tidak cocok dengan korpus; dibuat ulang.")

    print("Preprocessing teks untuk LDA (stemming, bisa lama)...")
    stemmer = StemmerFactory().create_stemmer()
    all_tokens = []
    started = time.time()

    for index, chunk in enumerate(chunks, start=1):
        all_tokens.append(
            preprocess_text(chunk["text"], stop_words, stemmer)
        )
        if index % 5000 == 0:
            print(
                f"  {index}/{len(chunks)} chunk "
                f"({time.time() - started:.0f} detik)"
            )

    temp_path = cfg.LDA_TOKENS_PATH.with_suffix(
        cfg.LDA_TOKENS_PATH.suffix + ".tmp"
    )
    with temp_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for tokens in all_tokens:
            file.write(json.dumps(tokens, ensure_ascii=False) + "\n")
    temp_path.replace(cfg.LDA_TOKENS_PATH)

    print(f"Cache token tersimpan: {cfg.LDA_TOKENS_PATH.name}")
    return all_tokens


# =========================================================
# SIDIK JARI EKSPERIMEN & PEMULIHAN
# =========================================================
def build_fingerprint(chunks_hash, stopwords_hash, dict_size, n_docs):
    return {
        "sample_name": cfg.SAMPLE_NAME,
        "chunks_file": cfg.CHUNKS_PATH.name,
        "chunks_sha256": chunks_hash,
        "stopwords_sha256": stopwords_hash,
        "preprocess_version": PREPROCESS_VERSION,
        "seed": SEED,
        "passes": PASSES,
        "iterations": ITERATIONS,
        "chunksize": CHUNKSIZE,
        "alpha": "symmetric",
        "eta": "symmetric",
        "no_below": NO_BELOW,
        "no_above": NO_ABOVE,
        "keep_n": KEEP_N,
        "dictionary_size": dict_size,
        "training_documents": n_docs,
    }


def model_is_valid(num_topics, run_id):
    """Model per K dipakai ulang hanya bila lengkap dan cocok."""
    model_path = cfg.lda_model_k_path(num_topics)
    meta_path = cfg.lda_model_k_meta_path(num_topics)

    if not model_path.exists() or not meta_path.exists():
        return False

    try:
        meta = load_json(meta_path)
    except (OSError, json.JSONDecodeError):
        return False

    return (
        meta.get("run_id") == run_id
        and meta.get("num_topics") == num_topics
    )


def load_saved_results(run_id):
    if not cfg.LDA_RESULTS_PATH.exists():
        return []

    results = load_json(cfg.LDA_RESULTS_PATH)

    if not isinstance(results, list):
        raise ValueError(
            f"Format hasil eksperimen tidak valid: {cfg.LDA_RESULTS_PATH}"
        )

    valid = []
    for result in results:
        k = result.get("num_topics")
        coherence = result.get("coherence_cv")

        if (
            result.get("run_id") == run_id
            and k in TOPIC_COUNTS
            and isinstance(coherence, (int, float))
            and np.isfinite(coherence)
            and model_is_valid(k, run_id)
        ):
            valid.append(result)

    discarded = len(results) - len(valid)
    if discarded:
        print(
            f"{discarded} hasil lama dibuang "
            f"(tidak cocok dengan korpus/konfigurasi saat ini)."
        )

    return valid


# =========================================================
# PROGRAM UTAMA
# =========================================================
def main():
    if not cfg.CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"File chunk tidak ditemukan: {cfg.CHUNKS_PATH}"
        )

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Sampel: {cfg.SAMPLE_NAME}")
    print("Memuat chunk...")
    chunks = load_chunks()
    print(f"Jumlah chunk awal: {len(chunks)}")

    chunks_hash = sha256_file(cfg.CHUNKS_PATH)
    stop_words = load_indonesian_stopwords()
    stopwords_hash = sha256_obj(sorted(stop_words))

    all_tokens = load_or_build_tokens(
        chunks, stop_words, chunks_hash, stopwords_hash
    )

    # Buang chunk yang kosong setelah preprocessing.
    valid_indices = [i for i, t in enumerate(all_tokens) if t]
    tokenized_texts = [all_tokens[i] for i in valid_indices]
    print(f"Chunk setelah preprocessing: {len(tokenized_texts)}")

    if not tokenized_texts:
        raise ValueError("Tidak ada teks tersisa setelah preprocessing.")

    print("Membangun dictionary dan corpus BoW...")
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(
        no_below=NO_BELOW,
        no_above=NO_ABOVE,
        keep_n=KEEP_N,
    )

    if len(dictionary) == 0:
        raise ValueError("Dictionary kosong setelah proses filter.")

    bow_corpus = [dictionary.doc2bow(t) for t in tokenized_texts]

    # Buang dokumen tanpa token yang ada di dictionary.
    nonempty = [i for i, bow in enumerate(bow_corpus) if bow]
    tokenized_texts = [tokenized_texts[i] for i in nonempty]
    bow_corpus = [bow_corpus[i] for i in nonempty]

    if not bow_corpus:
        raise ValueError("Corpus BoW kosong.")

    dictionary.save(str(cfg.LDA_DICTIONARY_PATH))

    fingerprint = build_fingerprint(
        chunks_hash, stopwords_hash, len(dictionary), len(bow_corpus)
    )
    run_id = sha256_obj(fingerprint)[:16]

    print(f"Ukuran dictionary: {len(dictionary)}")
    print(f"Dokumen untuk LDA: {len(bow_corpus)}")
    print(f"ID eksperimen   : {run_id}")

    # -----------------------------------------------------
    # Pulihkan hasil yang masih valid untuk korpus ini
    # -----------------------------------------------------
    results = load_saved_results(run_id)
    done = {r["num_topics"] for r in results}

    if done:
        print(f"K yang sudah selesai: {sorted(done)}")

    # -----------------------------------------------------
    # Eksperimen untuk setiap jumlah topik
    # -----------------------------------------------------
    for num_topics in TOPIC_COUNTS:
        if num_topics in done:
            print(f"\nK={num_topics} sudah selesai; dilewati.")
            continue

        model_path = cfg.lda_model_k_path(num_topics)
        meta_path = cfg.lda_model_k_meta_path(num_topics)

        if model_is_valid(num_topics, run_id):
            # Model selesai disimpan, tetapi C_v belum tercatat.
            print(
                f"\nModel K={num_topics} sudah ada; "
                f"hanya menghitung C_v..."
            )
            model = LdaModel.load(str(model_path))
        else:
            print(f"\nMelatih LDA dengan {num_topics} topik...")
            started = time.time()

            # Hapus penanda lama sebelum menulis model baru, agar
            # model setengah tertulis tidak dianggap valid.
            meta_path.unlink(missing_ok=True)

            model = LdaModel(
                corpus=bow_corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=SEED,
                passes=PASSES,
                iterations=ITERATIONS,
                alpha="symmetric",
                eta="symmetric",
                eval_every=None,
                chunksize=CHUNKSIZE,
                minimum_probability=0.0,
                per_word_topics=False,
            )

            # Simpan model SEBELUM menghitung C_v; penanda ditulis
            # terakhir sebagai tanda model lengkap.
            model.save(str(model_path))
            save_json_atomic(
                {
                    "run_id": run_id,
                    "num_topics": num_topics,
                    "fingerprint": fingerprint,
                },
                meta_path,
            )
            print(f"Pelatihan selesai ({time.time() - started:.0f} detik).")

        started = time.time()
        coherence_model = CoherenceModel(
            model=model,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
        )
        coherence = float(coherence_model.get_coherence())

        if not np.isfinite(coherence):
            raise ValueError(
                f"Coherence tidak valid untuk K={num_topics}: {coherence}"
            )

        results.append({
            "num_topics": num_topics,
            "coherence_cv": coherence,
            "seed": SEED,
            "passes": PASSES,
            "iterations": ITERATIONS,
            "run_id": run_id,
        })
        results.sort(key=lambda item: item["num_topics"])
        save_json_atomic(results, cfg.LDA_RESULTS_PATH)

        print(
            f"Coherence C_v: {coherence:.4f} "
            f"({time.time() - started:.0f} detik)"
        )
        print(f"Hasil K={num_topics} tersimpan.")

    # -----------------------------------------------------
    # Pilih K dan simpan model akhir
    # -----------------------------------------------------
    if not results:
        raise RuntimeError("Belum ada model LDA yang berhasil dilatih.")

    by_k = {r["num_topics"]: r for r in results}

    if SELECTED_K_OVERRIDE is not None:
        if SELECTED_K_OVERRIDE not in by_k:
            raise ValueError(
                f"SELECTED_K_OVERRIDE={SELECTED_K_OVERRIDE} "
                f"tidak ada di hasil: {sorted(by_k)}"
            )
        best_result = by_k[SELECTED_K_OVERRIDE]
        selection_basis = "manual_after_interpretability_review"
    else:
        best_result = max(results, key=lambda r: r["coherence_cv"])
        selection_basis = "highest_coherence_cv_candidate"

    best_k = best_result["num_topics"]
    best_score = best_result["coherence_cv"]
    best_model = LdaModel.load(str(cfg.lda_model_k_path(best_k)))
    best_model.save(str(cfg.LDA_MODEL_PATH))

    save_json_atomic(
        {
            "selected_num_topics_candidate": best_k,
            "coherence_cv": best_score,
            "seed": SEED,
            "passes": PASSES,
            "iterations": ITERATIONS,
            "chunksize": CHUNKSIZE,
            "no_below": NO_BELOW,
            "no_above": NO_ABOVE,
            "keep_n": KEEP_N,
            "dictionary_size": len(dictionary),
            "training_documents": len(bow_corpus),
            "preprocessing": [
                "portal_marker_removal",
                "lowercase",
                "remove_non_letters",
                "remove_stopwords",
                "Sastrawi_stemming",
            ],
            "extra_stopwords": sorted(EXTRA_STOPWORDS),
            "selection_basis": selection_basis,
            "sample_name": cfg.SAMPLE_NAME,
            "chunks_file": cfg.CHUNKS_PATH.name,
            "chunks_sha256": chunks_hash,
            "run_id": run_id,
            "gensim_version": gensim.__version__,
        },
        cfg.LDA_CONFIG_PATH,
    )

    # -----------------------------------------------------
    # Kata teratas untuk 3 K terbaik (bahan pemeriksaan
    # interpretabilitas manual untuk laporan)
    # -----------------------------------------------------
    top_results = sorted(
        results, key=lambda r: r["coherence_cv"], reverse=True
    )[:3]

    with cfg.LDA_TOPWORDS_PATH.open("w", encoding="utf-8") as file:
        for result in top_results:
            k = result["num_topics"]
            model = LdaModel.load(str(cfg.lda_model_k_path(k)))
            file.write(
                f"===== K={k} | C_v={result['coherence_cv']:.4f} =====\n"
            )
            for topic_id in range(k):
                words = ", ".join(
                    word for word, _ in model.show_topic(topic_id, topn=10)
                )
                file.write(f"Topik {topic_id}: {words}\n")
            file.write("\n")

    print("\n========== HASIL EKSPERIMEN LDA ==========")
    for result in results:
        marker = " <- dipilih" if result["num_topics"] == best_k else ""
        print(
            f"K={result['num_topics']:2d} | "
            f"C_v={result['coherence_cv']:.4f}{marker}"
        )

    print(f"\nDasar pemilihan: {selection_basis}")
    print(f"K terpilih     : {best_k}")
    print(f"Coherence      : {best_score:.4f}")
    print(f"Model akhir    : {cfg.LDA_MODEL_PATH}")
    print(f"Dictionary     : {cfg.LDA_DICTIONARY_PATH}")
    print(f"Hasil          : {cfg.LDA_RESULTS_PATH}")
    print(f"Konfigurasi    : {cfg.LDA_CONFIG_PATH}")
    print(f"Kata teratas   : {cfg.LDA_TOPWORDS_PATH}")

    print(f"\n10 kata utama pada setiap topik (K={best_k}):")
    for topic_id in range(best_k):
        print(f"Topik {topic_id}: {best_model.show_topic(topic_id, topn=10)}")


if __name__ == "__main__":
    main()