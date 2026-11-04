import json
import re
import random
from pathlib import Path

import numpy as np
import nltk
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# =========================================================
# KONFIGURASI
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
# CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks_sample_1000.jsonl"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

SEED = 42

TOPIC_COUNTS = [5, 10, 15, 20, 25, 30]

PASSES = 5
ITERATIONS = 100
NO_BELOW = 5
NO_ABOVE = 0.5
KEEP_N = 100_000

# UNTUK FINAL
# RESULTS_PATH = OUTPUT_DIR / "lda_coherence_results.json"
# DICTIONARY_PATH = MODEL_DIR / "lda_dictionary.dict"
# BEST_MODEL_PATH = MODEL_DIR / "lda_model.model"
# CONFIG_PATH = MODEL_DIR / "lda_config.json"

# UNTUK SAMPLE 1000
RESULTS_PATH = OUTPUT_DIR / "lda_coherence_results_sample_1000.json"
DICTIONARY_PATH = MODEL_DIR / "lda_dictionary_sample_1000.dict"
BEST_MODEL_PATH = MODEL_DIR / "lda_model_sample_1000.model"
CONFIG_PATH = MODEL_DIR / "lda_config_sample_1000.json"


random.seed(SEED)
np.random.seed(SEED)


# =========================================================
# PENYIMPANAN JSON YANG LEBIH AMAN
# =========================================================
def save_json_atomic(data, path):
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temp_path.replace(path)


def load_saved_results():
    if not RESULTS_PATH.exists():
        return []

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    if not isinstance(results, list):
        raise ValueError(
            f"Format hasil eksperimen tidak valid: {RESULTS_PATH}"
        )

    # Hanya gunakan hasil yang sesuai konfigurasi eksperimen ini.
    return [
        result for result in results
        if result.get("seed") == SEED
        and result.get("passes") == PASSES
        and result.get("iterations") == ITERATIONS
        and result.get("num_topics") in TOPIC_COUNTS
        and isinstance(result.get("coherence_cv"), (int, float))
        and np.isfinite(result["coherence_cv"])
        and (MODEL_DIR / f"lda_model_k{result['num_topics']}.model").exists()
    ]


# =========================================================
# MEMUAT DATA
# =========================================================
def load_chunks():
    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
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

    words.update({
        "cnn", "indonesia", "jakarta", "sumber",
        "foto", "video", "halaman", "berita"
    })

    return words


# =========================================================
# PREPROCESSING
# =========================================================
def preprocess_text(text, stop_words, stemmer):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [
        token for token in tokens
        if token not in stop_words and len(token) > 1
    ]

    return [stemmer.stem(token) for token in tokens]


# =========================================================
# PELATIHAN LDA
# =========================================================
def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"File chunk tidak ditemukan: {CHUNKS_PATH}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Memuat chunk...")
    chunks = load_chunks()

    # PILOT_SIZE = 5000
    # PILOT_SEED = 42

    # if len(chunks) > PILOT_SIZE:
    #     rng = random.Random(PILOT_SEED)
    #     chunks = rng.sample(chunks, PILOT_SIZE)


    texts = [chunk["text"] for chunk in chunks]

    print(f"Jumlah chunk awal: {len(texts)}")

    stop_words = load_indonesian_stopwords()
    stemmer = StemmerFactory().create_stemmer()

    print("Preprocessing teks untuk LDA...")
    tokenized_texts = [
        preprocess_text(text, stop_words, stemmer)
        for text in texts
    ]

    # Buang teks kosong setelah preprocessing.
    valid_indices = [
        i for i, tokens in enumerate(tokenized_texts) if tokens
    ]
    tokenized_texts = [tokenized_texts[i] for i in valid_indices]

    print(f"Chunk setelah preprocessing: {len(tokenized_texts)}")

    if not tokenized_texts:
        raise ValueError(
            "Tidak ada teks tersisa setelah preprocessing."
        )

    print("Membangun dictionary dan corpus BoW...")
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(
        no_below=NO_BELOW,
        no_above=NO_ABOVE,
        keep_n=KEEP_N,
    )

    if len(dictionary) == 0:
        raise ValueError("Dictionary kosong setelah proses filter.")

    bow_corpus = [
        dictionary.doc2bow(tokens)
        for tokens in tokenized_texts
    ]

    # Buang dokumen yang tidak memiliki token di dictionary.
    nonempty_indices = [
        i for i, bow in enumerate(bow_corpus) if bow
    ]
    tokenized_texts = [
        tokenized_texts[i] for i in nonempty_indices
    ]
    bow_corpus = [bow_corpus[i] for i in nonempty_indices]

    if not bow_corpus:
        raise ValueError("Corpus BoW kosong.")

    # Simpan dictionary untuk digunakan kembali.
    dictionary.save(str(DICTIONARY_PATH))

    print(f"Ukuran dictionary: {len(dictionary)}")
    print(f"Dokumen untuk LDA: {len(bow_corpus)}")

    # Muat hasil yang sudah selesai dan modelnya masih tersedia.
    experiment_results = load_saved_results()
    completed_topics = {
        result["num_topics"] for result in experiment_results
    }

    if completed_topics:
        print(f"Konfigurasi yang sudah tersimpan: {sorted(completed_topics)}")
        print("Konfigurasi tersebut akan dilewati.")

    # Tentukan model terbaik dari hasil tersimpan.
    best_result = (
        max(experiment_results, key=lambda item: item["coherence_cv"])
        if experiment_results else None
    )

    best_score = (
        best_result["coherence_cv"]
        if best_result else float("-inf")
    )
    best_k = (
        best_result["num_topics"]
        if best_result else None
    )
    best_model = (
        LdaModel.load(str(MODEL_DIR / f"lda_model_k{best_k}.model"))
        if best_k is not None else None
    )

    # -----------------------------------------------------
    # Eksperimen untuk setiap jumlah topik
    # -----------------------------------------------------
    for num_topics in TOPIC_COUNTS:
        if num_topics in completed_topics:
            print(f"\nK={num_topics} sudah tersimpan; dilewati.")
            continue

        print(f"\nMelatih LDA dengan {num_topics} topik...")

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
            chunksize=1000,
            minimum_probability=0.0,
            per_word_topics=False,
        )

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

        result = {
            "num_topics": num_topics,
            "coherence_cv": coherence,
            "seed": SEED,
            "passes": PASSES,
            "iterations": ITERATIONS,
        }

        # Simpan model setiap K agar dapat melanjutkan jika terhenti.
        individual_model_path = (
            MODEL_DIR / f"lda_model_k{num_topics}.model"
        )
        model.save(str(individual_model_path))

        experiment_results.append(result)
        experiment_results.sort(key=lambda item: item["num_topics"])

        # Simpan hasil segera setelah satu eksperimen selesai.
        save_json_atomic(experiment_results, RESULTS_PATH)

        print(f"Coherence C_v: {coherence:.4f}")
        print(f"Model K={num_topics} tersimpan.")
        print("Hasil eksperimen berhasil disimpan.")

        if coherence > best_score:
            best_score = coherence
            best_model = model
            best_k = num_topics
            best_result = result

            # Perbarui model terbaik sementara.
            best_model.save(str(BEST_MODEL_PATH))

            config = {
                "selected_num_topics_candidate": best_k,
                "coherence_cv": best_score,
                "seed": SEED,
                "passes": PASSES,
                "iterations": ITERATIONS,
                "no_below": NO_BELOW,
                "no_above": NO_ABOVE,
                "dictionary_size": len(dictionary),
                "training_documents": len(bow_corpus),
                "preprocessing": [
                    "lowercase",
                    "remove_non_letters",
                    "remove_stopwords",
                    "Sastrawi_stemming",
                ],
                "selection_basis": "highest_coherence_cv_candidate",
            }
            save_json_atomic(config, CONFIG_PATH)

    # -----------------------------------------------------
    # Pastikan model terbaik final tersimpan
    # -----------------------------------------------------
    if best_k is None or best_model is None:
        raise RuntimeError(
            "Belum ada model LDA yang berhasil dilatih atau dimuat."
        )

    best_model.save(str(BEST_MODEL_PATH))

    final_config = {
        "selected_num_topics_candidate": best_k,
        "coherence_cv": best_score,
        "seed": SEED,
        "passes": PASSES,
        "iterations": ITERATIONS,
        "no_below": NO_BELOW,
        "no_above": NO_ABOVE,
        "dictionary_size": len(dictionary),
        "training_documents": len(bow_corpus),
        "preprocessing": [
            "lowercase",
            "remove_non_letters",
            "remove_stopwords",
            "Sastrawi_stemming",
        ],
        "selection_basis": "highest_coherence_cv_candidate",
    }
    save_json_atomic(final_config, CONFIG_PATH)

    print("\n========== HASIL EKSPERIMEN LDA ==========")
    for result in sorted(
        experiment_results, key=lambda item: item["num_topics"]
    ):
        print(
            f"K={result['num_topics']:2d} | "
            f"C_v={result['coherence_cv']:.4f}"
        )

    print(f"\nKandidat berdasarkan C_v: {best_k}")
    print(f"Coherence terbaik: {best_score:.4f}")
    print(f"Model terbaik: {BEST_MODEL_PATH}")
    print(f"Dictionary: {DICTIONARY_PATH}")
    print(f"Hasil eksperimen: {RESULTS_PATH}")
    print(f"Konfigurasi: {CONFIG_PATH}")

    print("\n10 kata utama pada setiap topik model terbaik:")
    for topic_id in range(best_k):
        print(
            f"Topik {topic_id}: "
            f"{best_model.show_topic(topic_id, topn=10)}"
        )


if __name__ == "__main__":
    main()