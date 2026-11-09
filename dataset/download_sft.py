from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

RAW_DIR = Path("data/raw/sft")

# SmolTalk subsets that suit a small model. Skipped: math reasoning, code and
# long context (too hard or too long), system prompt chats (no system token),
# function calling (apigen-80k, for the tool use stage later)
SMOLTALK_SUBSETS = [
    "smol-magpie-ultra",
    "everyday-conversations",
    "smol-constraints",
    "smol-rewrite",
    "smol-summarize",
    "openhermes-100k",
    "explore-instruct-rewriting",
]

snapshot_download(
    repo_id="HuggingFaceTB/smoltalk",
    repo_type="dataset",
    allow_patterns=[f"data/{subset}/*.parquet" for subset in SMOLTALK_SUBSETS],
    local_dir=RAW_DIR / "smoltalk",
)

# Science tutoring dialogues, generated from the LibreTexts textbooks
snapshot_download(
    repo_id="princeton-nlp/TutorChat",
    repo_type="dataset",
    allow_patterns="data/*.parquet",
    local_dir=RAW_DIR / "tutorchat",
)

for name in ["smoltalk", "tutorchat"]:
    files = sorted((RAW_DIR / name).rglob("*.parquet"))
    n_rows = sum(pq.read_metadata(f).num_rows for f in files)
    n_bytes = sum(f.stat().st_size for f in files)
    print(f"{name}: {len(files)} files, {n_rows:,} rows, {n_bytes / 1e9:.1f} GB")
