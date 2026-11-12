import re

# Pola yang dihapus HANYA pada tahap preprocessing LDA (teks chunk dan embedding
# tidak diubah). Gunakan fungsi yang sama untuk pelatihan LDA, distribusi topik
# chunk, dan kueri.
POLA_PENANDA_PORTAL = [
    r"^\s*(?:[A-Za-z .]+,\s*)?CNN Indonesia\s*(?:-\s*)+",                              # Jakarta, CNN Indonesia - -
    r"^\s*(?:[A-Za-z .]+)?\(\s*ANTARA News\s*\)\s*[-\u2013]?",                         # Jakarta (ANTARA News) -
    r"^\s*(?:[A-Za-z .]+,\s*)?Merdeka\.com\s*[-\u2013]",                               # Merdeka.com -
    r"^\s*(?:[A-Za-z .]+,\s*)?JUARA\.net\s*(?:,\s*[A-Za-z .]+?)?\s*[-\u2013]",         # JUARA.net -, KUCHING, JUARA.net -, JUARA.NET, KUALA LUMPUR -
    r"^\s*Rimanews\s*[-\u2013]",                                                       # Rimanews -
    r"^\s*Suara\.com\s*[-\u2013]",                                                     # Suara.com -
    r"^\s*(?:[A-Za-z .]+)?\(\s*Pos Kota\s*\)\s*[-\u2013]",                             # JAKARTA (Pos Kota) –
    r"(?:/\s*)*<!\s*\[CDATA\[.*?\]\]>?",                                               # sisa skrip web (Goal Indonesia)
    r"Kamu bisa menjadi member komunitas startup dan inovasi DailySocial\.id.*$",      # penutup DailySocial
]
_POLA = [re.compile(p, re.IGNORECASE) for p in POLA_PENANDA_PORTAL]


def hapus_penanda_portal(teks):
    for pola in _POLA:
        teks = pola.sub(" ", teks)
    return teks.strip()

