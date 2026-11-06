"""
Konfigurasi bersama. Ganti SAMPLE_NAME untuk berpindah korpus;
semua path turunan mengikuti otomatis.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# "sample_1000" (pilot) atau "sample_10000" (eksperimen final)
SAMPLE_NAME = "sample_10000"

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

# Data
ARTICLES_PATH = DATA_DIR / f"articles_{SAMPLE_NAME}.jsonl"
CHUNKS_PATH = DATA_DIR / f"chunks_{SAMPLE_NAME}.jsonl"

# Embedding
EMBEDDINGS_PATH = MODEL_DIR / f"chunk_embeddings_{SAMPLE_NAME}.npy"
CHUNK_METADATA_PATH = MODEL_DIR / f"chunk_metadata_{SAMPLE_NAME}.jsonl"

# LDA
LDA_TOKENS_PATH = DATA_DIR / f"lda_tokens_{SAMPLE_NAME}.jsonl"
LDA_RESULTS_PATH = DATA_DIR / f"lda_coherence_results_{SAMPLE_NAME}.json"
LDA_TOPWORDS_PATH = DATA_DIR / f"lda_topwords_{SAMPLE_NAME}.txt"
LDA_DICTIONARY_PATH = MODEL_DIR / f"lda_dictionary_{SAMPLE_NAME}.dict"
LDA_MODEL_PATH = MODEL_DIR / f"lda_model_{SAMPLE_NAME}.model"
LDA_CONFIG_PATH = MODEL_DIR / f"lda_config_{SAMPLE_NAME}.json"

# Distribusi topik per chunk
TOPIC_DIST_PATH = MODEL_DIR / f"topic_distributions_{SAMPLE_NAME}.npy"
TOPIC_METADATA_PATH = MODEL_DIR / f"topic_metadata_{SAMPLE_NAME}.jsonl"


def lda_model_k_path(num_topics):
    """Model sementara per K (untuk melanjutkan proses)."""
    return MODEL_DIR / f"lda_model_k{num_topics}_{SAMPLE_NAME}.model"


def lda_model_k_meta_path(num_topics):
    """Penanda bahwa model per K selesai disimpan dan cocok dengan korpus."""
    return MODEL_DIR / f"lda_model_k{num_topics}_{SAMPLE_NAME}.meta.json"