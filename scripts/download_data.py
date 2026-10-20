"""Download SmallMinesDS from Hugging Face into data/.

Usage: .venv\\Scripts\\python scripts\\download_data.py
"""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "ellaampy/SmallMinesDS"
# Pinned dataset revision so every user gets the same files
REVISION = "0cb168b807206d9d34c0d1bd3c81097e69bb889b"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"

# Path in the dataset repo -> expected SHA-256 (None: not checked)
FILES = {
    "SmallMinesDS.zip": "829e24dfc84fb5a188677587fea90159d9f3721f9d5861dbe005f09013ec84c6",
    "data_splits/train_test_splits_2016.csv": None,
    "data_splits/train_test_splits_2022.csv": None,
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download(path: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(f"{BASE_URL}/{path}") as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {path}: {done / 2**20:.0f}/{total / 2**20:.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    for path, expected in FILES.items():
        dest = DATA_DIR / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"  {path}: already present")
        else:
            download(path, dest)
        if expected and sha256(dest) != expected:
            print(f"FAIL: checksum mismatch for {path}; delete it and run again.")
            return 1

    # The zip has a top-level SmallMinesDS/ folder, so this yields data/SmallMinesDS/<year>/
    if not (DATA_DIR / "SmallMinesDS").exists():
        print("Extracting SmallMinesDS.zip...", flush=True)
        with zipfile.ZipFile(DATA_DIR / "SmallMinesDS.zip") as z:
            z.extractall(DATA_DIR)
    print("OK: data ready in", DATA_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
