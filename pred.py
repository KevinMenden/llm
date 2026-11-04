# %%

import torch
from tokenizers import Tokenizer

from llm import LLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
weights_path = (
    "/home/kevin/repos/llm/logs/checkpoints/model_large_100m_rope_long_9000.ckpt"
)
checkpoint = torch.load(weights_path, weights_only=True)
config = checkpoint["config"]

tok = Tokenizer.from_file("tokenizer.json")
model = LLM(
    vocab_size=tok.get_vocab_size(),
    n_dim=config.n_dim,
    n_heads=config.n_heads,
    n_layers=config.n_layers,
).to(DEVICE)
model.eval()


model.load_state_dict(checkpoint["model_state_dict"])


# Generate a few tokens
n_tokens = 100
text = "An elephant is"
ids = torch.tensor(tok.encode(text).ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

model.eval()
for i in range(n_tokens):
    ids = model.generate(ids)

new_text = tok.decode(ids[0].tolist())
print(new_text)
