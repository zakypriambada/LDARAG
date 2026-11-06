import platform, os, sys
from importlib import metadata

print("OS        :", platform.platform())
print("Prosesor  :", platform.processor() or platform.machine())
print("Jumlah core (logis):", os.cpu_count())
try:
    import psutil
    print("RAM total : %.1f GB" % (psutil.virtual_memory().total / 1024**3))
except ImportError:
    print("RAM total : (pasang psutil: pip install psutil)")
try:
    import torch
    print("GPU       :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "tidak ada / tidak dipakai (CPU)")
except ImportError:
    print("GPU       : (torch tidak terpasang)")
print("Python    :", sys.version.split()[0])
for p in ["numpy", "scipy", "gensim", "nltk", "Sastrawi", "sentence-transformers",
          "torch", "chromadb", "google-genai"]:
    try:
        print(f"{p:22s}:", metadata.version(p))
    except metadata.PackageNotFoundError:
        print(f"{p:22s}: tidak terpasang")