import argparse
import glob
import os

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from llm import LLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path: str, vocab_size: int) -> tuple[LLM, int]:
    """Returns the model and the block size it was trained with"""
    # mmap, so the optimizer state in the checkpoint doesn't get loaded into memory
    checkpoint = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    config = checkpoint["config"]
    model = LLM(
        vocab_size=vocab_size,
        n_dim=config["n_dim"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        context_size=config["context_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Loaded {path}: step {checkpoint['step']}, "
        f"{checkpoint['tokens_seen']:,} tokens seen, val loss {checkpoint['val_loss']}"
    )
    print(
        f"Context: {config['context_size']} tokens, "
        f"trained on sequences of {config['block_size']} tokens"
    )
    return model.to(DEVICE).eval(), config["block_size"]


def context_info(n_prompt: int, n_reply: int, context_size: int, block_size: int) -> str:
    """One line summary of how much of the context is used"""
    total = n_prompt + n_reply
    info = f"[{total}/{context_size} tokens: {n_prompt} prompt + {n_reply} reply"
    if total > context_size:
        info += f", oldest {total - context_size} dropped"
    elif total > block_size:
        info += f", past the {block_size} tokens seen in training"
    return info + "]"


def sample(logits, temperature: float, top_k: int, top_p: float) -> int:
    """Sample a token from logits of shape (vocab_size,)"""
    if temperature == 0:
        return logits.argmax().item()
    logits = logits.float() / temperature

    if top_k > 0:
        # everything below the k-th largest logit can't be sampled
        threshold = logits.topk(min(top_k, logits.numel())).values[-1]
        logits[logits < threshold] = float("-inf")

    if top_p < 1.0:
        # keep the most likely tokens until their probabilities add up to top_p
        sorted_logits, sorted_ids = logits.sort(descending=True)
        cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()  # keep the token that crosses top_p
        remove[0] = False
        logits[sorted_ids[remove]] = float("-inf")

    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).item()


@torch.no_grad()
def generate(model: LLM, tok: Tokenizer, ids: list[int], stop_ids: set[int], args) -> list[int]:
    """Generate and print tokens until a stop token or max_tokens, returns the new ids"""
    new_ids = []
    printed = ""
    for _ in range(args.max_tokens):
        context = torch.tensor([(ids + new_ids)[-model.context_size :]], device=DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            logits, _ = model(context)
        token = sample(logits[0, -1], args.temperature, args.top_k, args.top_p)
        if token in stop_ids:
            break
        new_ids.append(token)

        # a character can span several tokens, wait until it's complete before printing
        text = tok.decode(new_ids)
        if not text.endswith("�"):
            print(text[len(printed) :], end="", flush=True)
            printed = text
    print()
    return new_ids


def main():
    parser = argparse.ArgumentParser(description="Talk to a trained model")
    parser.add_argument("--checkpoint", help="defaults to the newest checkpoint")
    parser.add_argument("--tokenizer", default="dataset/tokenizer.json")
    parser.add_argument(
        "--mode",
        choices=["complete", "chat"],
        default="complete",
        help="complete: continue the text. chat: use the chat format, needs an SFT model",
    )
    parser.add_argument("--prompt", help="run a single prompt and exit")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 is greedy")
    parser.add_argument("--top-k", type=int, default=50, help="0 turns it off")
    parser.add_argument("--top-p", type=float, default=0.95, help="1 turns it off")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    path = args.checkpoint or max(glob.glob("logs/checkpoints/*.ckpt"), key=os.path.getmtime)
    tok = Tokenizer.from_file(args.tokenizer)
    model, block_size = load_model(path, tok.get_vocab_size())

    eot = tok.token_to_id("<|endoftext|>")
    end = tok.token_to_id("<|end|>")
    history = []  # only used in chat mode

    def respond(prompt: str):
        nonlocal history
        if args.mode == "chat":
            # the prompt includes the whole conversation so far
            history += tok.encode(f"<|user|>{prompt}<|end|><|assistant|>").ids
            n_prompt = len(history)
            reply = generate(model, tok, history, {eot, end}, args)
            history += reply + [end]
        else:
            # EOT marks the start of a new document, like in the training data
            ids = [eot] + tok.encode(prompt).ids
            n_prompt = len(ids)
            reply = generate(model, tok, ids, {eot}, args)
        print(context_info(n_prompt, len(reply), model.context_size, block_size))

    if args.prompt is not None:
        respond(args.prompt)
        return

    print(f"Mode: {args.mode}. Ctrl-D quits, ctrl-C stops a reply, /reset clears the chat.")
    while True:
        try:
            prompt = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt == "/reset":
            history = []
            continue
        try:
            respond(prompt)
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
