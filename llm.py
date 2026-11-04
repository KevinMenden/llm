# %%
import math

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import Tensor, nn

def precompute_rope_tables(context_size: int, d_head: int, base: float= 10_000.0):
    """Precompute cos/sin tables of shape (context_size, d_head // 2)"""
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    positions = torch.arange(context_size).float()
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()

def rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x_f32 = x.float() # rotate in fp32
    x1, x2 = x_f32.chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.type_as(x)



class CausalSelfAttention(nn.Module):
    """Causal self attention module"""

    def __init__(self, n_dim: int = 128, n_heads: int = 4):
        super().__init__()

        self.q = nn.Linear(n_dim, n_dim, bias=False)
        self.k = nn.Linear(n_dim, n_dim, bias=False)
        self.v = nn.Linear(n_dim, n_dim, bias=False)
        self.proj = nn.Linear(n_dim, n_dim, bias=False)
        self.d_head = n_dim // n_heads
        self.n_heads = n_heads

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        # Q, K, V with dimensions B, n_head, T, d_head
        queries = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        keys = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        values = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # apply RoPE
        queries = rope(queries, cos, sin)
        keys = rope(keys, cos, sin)

        # use pytorch scaled_dot_product_attention instead, it's twice as fast ...
        attention_result = F.scaled_dot_product_attention(queries, keys, values, is_causal=True)

        out = attention_result.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, n_dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(n_dim, hidden_dim, bias=False)
        self.value = nn.Linear(n_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, n_dim, bias=False)

    def forward(self, x):
        return self.out(F.silu(self.gate(x)) * self.value(x))


class Block(nn.Module):
    """One block of the model"""

    def __init__(
        self,
        n_dim: int = 128,
        n_heads: int = 4,
        expansion_factor: int = 4,
    ) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.ln1 = nn.RMSNorm(n_dim)
        self.ln2 = nn.RMSNorm(n_dim)
        self.attn = CausalSelfAttention(
            n_dim=n_dim, n_heads=n_heads
        )
        hidden_dim = int(2 * expansion_factor * n_dim / 3)
        self.feed_forward = SwiGLU(n_dim, hidden_dim)

    def forward(self, x, cos, sin):
        # layer norm, attention and residual connection
        x = x + self.attn(self.ln1(x), cos, sin)
        # then layer norm and feed forward layers
        x = x + self.feed_forward(self.ln2(x))
        return x


class LLM(nn.Module):
    """LLM module"""

    def __init__(
        self,
        n_dim: int = 128,
        context_size: int = 1024,
        n_heads: int = 4,
        n_layers: int = 4,
        vocab_size: int = 2000,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_dim)
        self.layers = nn.ModuleList(
            [
                Block(n_dim=n_dim, context_size=context_size, n_heads=n_heads)
                for _ in range(n_layers)
            ]
        )
        self.ln_final = nn.RMSNorm(n_dim)
        self.linear_out = nn.Linear(n_dim, vocab_size, bias=False)
        # weight tying: input embedding and output projection share one matrix
        self.linear_out.weight = self.tok_emb.weight
        self.context_size = context_size

        cos, sin = precompute_rope_tables(context_size=self.context_size, d_head = n_dim // n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scale down the layers that write into the residual stream (2 per block),
        # so its variance doesn't grow with depth
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("feed_forward.out.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 style init: N(0, 0.02) for linear/embedding weights, zero biases"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        T = ids.shape[1]
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.tok_emb(ids)

        for layer in self.layers:
            x = layer(x, cos, sin)

        logits = self.linear_out(self.ln_final(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, ids: Tensor, n_tokens: int = 1, temperature: float = 1.0) -> Tensor:
        """Generate new tokens"""
        ids_context = ids[:, -self.context_size:]
        logits, _ = self(ids_context)
        current_pos = logits[:, -1, :]
        probs = F.softmax(current_pos / temperature, dim=-1)

        sample_token = torch.multinomial(probs, num_samples=1)

        new_ids = torch.cat([ids, sample_token], dim=1)

        return new_ids


if __name__ == "__main__":

    # Config
    vocab_size = 2000
    n_dim = 128
    n_heads = 4
    n_layers = 4
    context_size = 1024

    # Run some tests
    model = LLM()
    tok = Tokenizer.from_file("tokenizer.json")
    text = "this is a test"
    tokens = tok.encode(text)

    ids = torch.tensor(tokens.ids[:-1], dtype=torch.long).unsqueeze(0)
    targets = torch.tensor(tokens.ids[1:], dtype=torch.long).unsqueeze(0)
    print(ids.shape)
    print(targets.shape)

    res, loss = model(ids, targets)
    print(res.shape)

    print(model.generate(ids))
