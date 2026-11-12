import json
import re
from pathlib import Path

import numpy as np
import nltk
from nltk.corpus import stopwords
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from penanda_portal import hapus_penanda_portal

ROOT = Path.cwd()
CHUNKS_PATH = ROOT / "data/processed/chunks_sample_1000.jsonl"
MODEL_PATH = ROOT / "models/lda_model_sample_1000.model"
DICT_PATH = ROOT / "models/lda_dictionary_sample_1000.dict"
CONFIG_PATH = ROOT / "models/lda_config_sample_1000.json"
DIST_PATH = ROOT / "models/topic_distributions_sample_1000.npy"
META_PATH = ROOT / "models/topic_metadata_sample_1000.jsonl"

# Muat stopword dengan cara yang sama seperti training
try:
    stop_words = set(stopwords.words("indonesian"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("indonesian"))

stop_words.update({
    "cnn", "indonesia", "jakarta", "sumber",
    "foto", "video", "halaman", "berita"
})

stemmer = StemmerFactory().create_stemmer()

def preprocess_text(text):
    text = hapus_penanda_portal(str(text))
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [
        token for token in tokens
        if token not in stop_words and len(token) > 1
    ]
    return [stemmer.stem(token) for token in tokens]

# Baca chunk dengan urutan aslinya
chunks = []
with CHUNKS_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

print(f"Jumlah chunk: {len(chunks)}")

# Muat model dan dictionary yang sudah dilatih
lda = LdaModel.load(str(MODEL_PATH))
dictionary = Dictionary.load(str(DICT_PATH))

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = json.load(f)

num_topics = int(config["selected_num_topics_candidate"])
if lda.num_topics != num_topics:
    raise ValueError(
        f"Jumlah topik model ({lda.num_topics}) tidak cocok "
        f"dengan konfigurasi ({num_topics})."
    )

if len(dictionary) != len(lda.id2word):
    raise ValueError("Ukuran dictionary tidak cocok dengan model LDA.")

distributions = []
metadata = []
empty_count = 0

for i, chunk in enumerate(chunks):
    text = chunk.get("text", "")
    tokens = preprocess_text(text)
    bow = dictionary.doc2bow(tokens)

    if bow:
        topic_dist = lda.get_document_topics(
            bow, minimum_probability=0.0
        )
        probs = np.zeros(num_topics, dtype=np.float32)
        for topic_id, probability in topic_dist:
            probs[topic_id] = probability
    else:
        # Distribusi seragam untuk chunk tanpa kata dalam dictionary
        probs = np.full(
            num_topics, 1.0 / num_topics, dtype=np.float32
        )
        empty_count += 1

    distributions.append(probs)
    metadata.append({
        "chunk_id": chunk["chunk_id"],
        "article_id": chunk["article_id"],
        "chunk_index": chunk.get("chunk_index", i)
    })

matrix = np.vstack(distributions).astype(np.float32)

# Validasi sebelum menyimpan
if matrix.shape != (len(chunks), num_topics):
    raise ValueError(f"Bentuk matriks tidak sesuai: {matrix.shape}")

if not np.isfinite(matrix).all():
    raise ValueError("Matriks berisi NaN atau nilai tak hingga.")

row_sums = matrix.sum(axis=1)
if not np.allclose(row_sums, 1.0, atol=1e-4):
    raise ValueError("Ada distribusi topik yang jumlah probabilitasnya bukan 1.")

# Simpan dengan urutan yang sama dengan chunks_sample_1000.jsonl
np.save(DIST_PATH, matrix)
with META_PATH.open("w", encoding="utf-8") as f:
    for row in metadata:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("\nBERHASIL")
print(f"Model topik       : {lda.num_topics}")
print(f"Ukuran dictionary : {len(dictionary)}")
print(f"Shape distribusi  : {matrix.shape}")
print(f"Chunk tanpa BoW   : {empty_count}")
print(f"Jumlah baris      : {len(metadata)}")
print(f"Jumlah unik ID    : {len(set(x['chunk_id'] for x in metadata))}")
print(f"Jumlah probabilitas per baris: min={row_sums.min():.6f}, max={row_sums.max():.6f}")
print(f"File distribusi   : {DIST_PATH}")
print(f"File metadata     : {META_PATH}")
