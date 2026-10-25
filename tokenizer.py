from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB_SIZE = 2000

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]

def data_iter():
    for i in range(0, len(ds), 1000):
        yield ds[i : i + 1000]["text"]


trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<|endoftext|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()

tok.train_from_iterator(data_iter(), trainer=trainer, length=len(ds))
tok.save("tokenizer.json")

print("Saved tokenizer")


# testing
tok = Tokenizer.from_file("tokenizer.json")
ids = tok.encode("cheers mate").ids
print(ids)
text = tok.decode(ids)
print(text)



