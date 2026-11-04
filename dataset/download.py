from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

RAW_DIR = Path("data/raw")

# LibreTexts chapters: 3 parquet files, about 0.5 GB
snapshot_download(
    repo_id="princeton-nlp/TextbookChapters",
    repo_type="dataset",
    allow_patterns="data/*.parquet",
    local_dir=RAW_DIR / "textbooks",
)

# FineWeb-Edu 10B token sample: 14 parquet files, 28.5 GB
snapshot_download(
    repo_id="HuggingFaceFW/fineweb-edu",
    repo_type="dataset",
    allow_patterns="sample/10BT/*.parquet",
    local_dir=RAW_DIR / "fineweb"
)



for name in ["fineweb", "textbooks"]:
    files = sorted((RAW_DIR / name).rglob("*.parquet"))
    n_rows = sum(pq.read_metadata(f).num_rows for f in files)
    n_bytes = sum(f.stat().st_size for f in files)
    print(f"{name}: {len(files)} files, {n_rows:,} rows, {n_bytes / 1e9:.1f} GB")
