from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import pyarrow.parquet as pq

VOCAB_SIZE = 16384
RAW_DATA_DIR = Path("data/raw")

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|pad|>",
    "<|reserved_0|>",
    "<|reserved_1|>",
    "<|reserved_2|>",
]

def read_texts(path: Path, column: str, batch_size: int = 1000):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=[column]):
        yield batch.column(column).to_pylist()

def data_iter():
    # one file is enough for the vocabulary
    yield from read_texts(RAW_DATA_DIR / "fineweb/sample/10BT/000_00000.parquet", "text")
    for path in sorted((RAW_DATA_DIR / "textbooks/data").glob("*.parquet")):
        yield from read_texts(path, "chapter")

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()

tok.train_from_iterator(data_iter(), trainer=trainer)
tok.save("dataset/tokenizer.json")

print("Saved tokenizer")


# Testing
tok = Tokenizer.from_file("dataset/tokenizer.json")
print("vocab size:", tok.get_vocab_size())
print({t: tok.token_to_id(t) for t in SPECIAL_TOKENS})

text = "<|user|>What does mitochondrial DNA encode?<|end|>\nCafé: 3 µm"
ids = tok.encode(text).ids
print(ids)
print([tok.decode([i], skip_special_tokens=False) for i in ids])
print(tok.decode(ids, skip_special_tokens=False))
