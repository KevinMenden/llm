import zlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data")
VAL_BOOK_PERCENT = 3
SKIP_SUBJECTS = {"human", "socialsci", "biz", "workforce"}

tok = Tokenizer.from_file("dataset/tokenizer.json")
EOT = tok.token_to_id("<|endoftext|>")
SPECIAL_TOKENS = [t.content for t in tok.get_added_tokens_decoder().values()]

def clean(text: str) -> str:
    # Remove any special tokens in the text
    if "<|" in text:
        for token in SPECIAL_TOKENS:
            text = text.replace(token, "")
    return text

def write_batch(texts: list[str], out) -> int:
    encodings = tok.encode_batch_fast([clean(t) for t in texts])
    ids = []
    for enc in encodings:
        ids.extend(enc.ids)
        ids.append(EOT)
    np.array(ids, dtype=np.uint16).tofile(out)
    return len(ids)

def prepare_fineweb():
    files = sorted((RAW_DIR / "fineweb/sample/10BT").glob("*.parquet"))
    splits = {"train": files[:-1], "val": files[-1:]}

    for split, split_files in splits.items():
        n_tokens = 0
        with open(OUT_DIR / f"fineweb_{split}.bin", "wb") as out:
            for path in split_files:
                parquet = pq.ParquetFile(path)
                for batch in parquet.iter_batches(batch_size=1000, columns=["text"]):
                    n_tokens += write_batch(batch.column("text").to_pylist(), out)
                    print(f"fineweb {split}: {path.name}, {n_tokens:,} tokens so far")
            print(f"fineweb {split}: {n_tokens:,} tokens")


def is_val_book(path: str) -> bool:
    """Whole books go to one split, so val only contains unseen books"""
    book = "/".join(path.split("/")[:4])
    return zlib.crc32(book.encode()) % 100 < VAL_BOOK_PERCENT


def prepare_textbooks():
    n_tokens = {"train": 0, "val": 0}
    with (
        open(OUT_DIR / "textbooks_train.bin", "wb") as train_out,
        open(OUT_DIR / "textbooks_val.bin", "wb") as val_out,
    ):
        for path in sorted((RAW_DIR / "textbooks/data").glob("*.parquet")):
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=1000):
                train_texts, val_texts = [], []
                chapters = batch.column("chapter").to_pylist()
                paths = batch.column("path").to_pylist()
                for chapter, chapter_path in zip(chapters, paths):
                    if chapter_path.split("/")[1] in SKIP_SUBJECTS:
                        continue
                    if is_val_book(chapter_path):
                        val_texts.append(chapter)
                    else:
                        train_texts.append(chapter)
                n_tokens["train"] += write_batch(train_texts, train_out)
                n_tokens["val"] += write_batch(val_texts, val_out)
    print(f"textbooks: {n_tokens}")

if __name__ == "__main__":
    prepare_textbooks()
    prepare_fineweb()

    # Check the result
    for name in ["textbooks_train", "textbooks_val", "fineweb_val", "fineweb_train"]:
        data = np.memmap(OUT_DIR / f"{name}.bin", dtype=np.uint16, mode="r")
        print(f"{name}: {len(data):,} tokens, {data[:1_000_000].max()} max id")
        print(repr(tok.decode(data[:50].tolist(), skip_special_tokens=False)))
