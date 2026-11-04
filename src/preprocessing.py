
import re


def flatten_tokens(data):
    """Mengubah token bertingkat menjadi teks biasa."""
    tokens = []

    def collect(item):
        if isinstance(item, list):
            for child in item:
                collect(child)
        elif isinstance(item, str):
            tokens.append(item)

    collect(data)
    return " ".join(tokens)


def clean_text(text):
    """Merapikan spasi dan tanda baca yang terpisah."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Merapatkan tanda baca yang terpisah dari kata sebelumnya
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)

    # Merapikan spasi setelah tanda kurung buka
    text = re.sub(r"([(\\[])\s+", r"\1", text)

    return text


def process_article(article):
    """Mengubah satu record IndoSum menjadi dokumen terstruktur."""
    article_text = clean_text(
        flatten_tokens(article.get("paragraphs", []))
    )
    summary_text = clean_text(
        flatten_tokens(article.get("summary", []))
    )

    return {
        "id": article.get("id"),
        "category": article.get("category"),
        "source": article.get("source"),
        "text": article_text,
        "summary": summary_text,
    }
