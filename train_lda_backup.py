
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
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

SEED = 42

# Uji beberapa jumlah topik. Bisa diperluas setelah pilot.
TOPIC_COUNTS = [5, 10, 15, 20, 25, 30]

PASSES = 5
ITERATIONS = 100
NO_BELOW = 5
NO_ABOVE = 0.5
KEEP_N = 100_000

random.seed(SEED)
np.random.seed(SEED)


def load_chunks():
    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def load_indonesian_stopwords():
    # Stopword dari NLTK ditambah kata umum bahasa Indonesia.
    try:
        words = set(stopwords.words("indonesian"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = set(stopwords.words("indonesian"))

    # Tambahan ini dapat disesuaikan berdasarkan inspeksi korpus.
    words.update({
        "cnn", "indonesia", "jakarta", "sumber",
        "foto", "video", "halaman", "berita"
    })

    return words


def preprocess_text(text, stop_words, stemmer):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [
        token for token in tokens
        if token not in stop_words and len(token) > 1
    ]

    # Stemming Sastrawi
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {CHUNKS_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Memuat chunk...")
    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]

    print(f"Jumlah chunk: {len(texts)}")

    stop_words = load_indonesian_stopwords()
    stemmer = StemmerFactory().create_stemmer()

    print("Preprocessing teks untuk LDA...")
    tokenized_texts = [
        preprocess_text(text, stop_words, stemmer)
        for text in texts
    ]

    # Buang dokumen yang kosong setelah preprocessing.
    valid_indices = [
        i for i, tokens in enumerate(tokenized_texts) if tokens
    ]
    tokenized_texts = [tokenized_texts[i] for i in valid_indices]
    valid_chunks = [chunks[i] for i in valid_indices]

    print(f"Chunk setelah preprocessing: {len(tokenized_texts)}")

    if not tokenized_texts:
        raise ValueError("Tidak ada teks yang tersisa setelah preprocessing.")

    print("Membangun dictionary dan corpus BoW...")
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(
        no_below=NO_BELOW,
        no_above=NO_ABOVE,
        keep_n=KEEP_N,
    )

    # Buang token yang tidak ada di dictionary hasil filter.
    bow_corpus = [
        dictionary.doc2bow(tokens)
        for tokens in tokenized_texts
    ]

    nonempty_indices = [
        i for i, bow in enumerate(bow_corpus) if bow
    ]
    tokenized_texts = [tokenized_texts[i] for i in nonempty_indices]
    bow_corpus = [bow_corpus[i] for i in nonempty_indices]
    valid_chunks = [valid_chunks[i] for i in nonempty_indices]

    if len(dictionary) == 0 or not bow_corpus:
        raise ValueError("Dictionary atau corpus BoW kosong.")

    dictionary.save(str(MODEL_DIR / "lda_dictionary.dict"))

    print(f"Ukuran dictionary: {len(dictionary)}")
    print(f"Dokumen untuk LDA: {len(bow_corpus)}")

    experiment_results = []
    best_score = float("-inf")
    best_model = None
    best_k = None

    for num_topics in TOPIC_COUNTS:
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

        result = {
            "num_topics": num_topics,
            "coherence_cv": coherence,
            "seed": SEED,
            "passes": PASSES,
            "iterations": ITERATIONS,
        }
        experiment_results.append(result)

        print(f"Coherence C_v: {coherence:.4f}")

        if coherence > best_score:
            best_score = coherence
            best_model = model
            best_k = num_topics

    # Simpan hasil seluruh eksperimen.
    results_path = OUTPUT_DIR / "lda_coherence_results.json"
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(experiment_results, file, ensure_ascii=False, indent=2)

    # Simpan model dengan coherence tertinggi sebagai kandidat awal.
    # Keputusan akhir tetap perlu mempertimbangkan interpretabilitas.
    best_model_path = MODEL_DIR / "lda_model.model"
    best_model.save(str(best_model_path))

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
    }

    with (MODEL_DIR / "lda_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print("\n========== HASIL EKSPERIMEN LDA ==========")
    for result in experiment_results:
        print(
            f"K={result['num_topics']:2d} | "
            f"C_v={result['coherence_cv']:.4f}"
        )

    print(f"\nKandidat berdasarkan C_v: {best_k}")
    print(f"Coherence terbaik: {best_score:.4f}")
    print(f"Model tersimpan: {best_model_path}")
    print(f"Hasil eksperimen: {results_path}")

    print("\n10 kata utama pada setiap topik kandidat:")
    for topic_id in range(best_k):
        print(f"Topik {topic_id}: {best_model.show_topic(topic_id, topn=10)}")


if __name__ == "__main__":
    main()
