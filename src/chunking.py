
def create_chunks(text, chunk_size=100, overlap=20):
    """Membagi teks menjadi chunk berdasarkan jumlah kata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size harus lebih dari 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap harus >= 0 dan < chunk_size")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks
