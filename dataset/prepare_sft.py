"""Tokenize the SFT conversations into flat arrays.

Per split, this writes to data/sft/:
- {split}_ids.npy: all conversations concatenated (uint16)
- {split}_mask.npy: 1 where the token is a training target, the assistant replies
  and their closing <|end|> (uint8)
- {split}_offsets.npy: start of each conversation, plus the total length at the end (int64)
- {split}_sources.npy: index into sources.json per conversation (uint8)
"""

import json
import re
import urllib.parse
import zlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from prepare import SKIP_SUBJECTS, clean, is_val_book, tok

RAW_DIR = Path("data/raw/sft")
OUT_DIR = Path("data/sft")
MAX_LEN = 1024

USER = tok.token_to_id("<|user|>")
ASSISTANT = tok.token_to_id("<|assistant|>")
END = tok.token_to_id("<|end|>")
EOT = tok.token_to_id("<|endoftext|>")

# TutorChat conversations look like "<s>\nuser: ...</s>\nassistant: ...</s>"
TUTORCHAT_TURN = re.compile(r"\n(user|assistant|question|answer): (.*?)</s>", re.S)

# Teacher-started TutorChat lessons get a user turn in front, so every conversation
# starts with the user, like in chat.py
OPENERS = [
    "Can you teach me about {}?",
    "I'd like to learn about {}.",
    "Could you explain {} to me?",
    "Let's go over {}.",
]


def chapter_title(path: str) -> str:
    """'.../8.2%3A_The_Light-Dependent_Reactions.txt' -> 'The Light-Dependent Reactions'"""
    name = urllib.parse.unquote(path.split("/")[-1]).removesuffix(".txt")
    if ":" in name:
        name = name.split(":", 1)[1]
    return name.replace("_", " ").strip()


def tutorchat_conversations(row: dict, split: str) -> list[list[tuple[str, str]]]:
    """Turn one TutorChat row into conversations of (role, content) turns"""
    text = row["processed_conversation"]
    path = row["textbook_folder"]
    # openbook dialogues start with the chapter text and refer to it
    if row["mode"] == "openbook" or not text.startswith("<s>"):
        return []
    if path.split("/")[1] in SKIP_SUBJECTS:
        return []
    # the textbook val books must stay unseen
    if split == "train" and is_val_book(path):
        return []

    turns = TUTORCHAT_TURN.findall(text)
    if row["name"] == "generateexam":
        # independent exam questions, with inconsistent role labels, so split them
        # into one question and answer pair each
        contents = [content for _, content in turns]
        return [
            [("user", question), ("assistant", answer)]
            for question, answer in zip(contents[0::2], contents[1::2])
        ]

    if turns and turns[0][0] == "assistant":
        opener = OPENERS[zlib.crc32(path.encode()) % len(OPENERS)]
        turns.insert(0, ("user", opener.format(chapter_title(path))))
    return [turns]


def smoltalk_conversation(messages: list[dict]) -> list[tuple[str, str]]:
    """SmolTalk messages as (role, content) turns, with system prompts merged into the
    following user turn, since there's no system token"""
    turns = []
    system = None
    for message in messages:
        if message["role"] == "system":
            system = message["content"]
        elif message["role"] == "user" and system is not None:
            turns.append(("user", f"{system}\n\n{message['content']}"))
            system = None
        else:
            turns.append((message["role"], message["content"]))
    return turns


def encode(conversations: list[list[tuple[str, str]]]):
    """Tokenize conversations, yields (ids, mask, truncated) per kept conversation"""
    contents = [clean(content.strip()) for turns in conversations for _, content in turns]
    encoded = iter(tok.encode_batch_fast(contents))

    for turns in conversations:
        ids, mask = [], []
        full = False
        for role, _ in turns:
            content_ids = next(encoded).ids  # always advance, to stay in sync
            # keep whole turns only, until one doesn't fit (the closing EOT included)
            if full or len(ids) + len(content_ids) + 3 > MAX_LEN:
                full = True
                continue
            if role == "assistant":
                ids += [ASSISTANT] + content_ids + [END]
                mask += [0] + [1] * (len(content_ids) + 1)
            else:
                ids += [USER] + content_ids + [END]
                mask += [0] * (len(content_ids) + 2)

        # cut after the last assistant turn, trailing user turns teach nothing
        if 1 not in mask:
            continue
        last = len(mask) - mask[::-1].index(1)
        truncated = full or last < len(ids)
        yield ids[:last] + [EOT], mask[:last] + [0], truncated


def iter_sources(split: str):
    """Yields (source name, list of conversations) in batches"""
    smoltalk_split = "train" if split == "train" else "test"
    for subset_dir in sorted((RAW_DIR / "smoltalk/data").iterdir()):
        for path in sorted(subset_dir.glob(f"{smoltalk_split}-*.parquet")):
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=2000, columns=["messages"]):
                conversations = [smoltalk_conversation(m) for m in batch.column("messages").to_pylist()]
                yield f"smoltalk/{subset_dir.name}", conversations

    tutorchat_split = "train" if split == "train" else "validation"
    for path in sorted((RAW_DIR / "tutorchat/data").glob(f"{tutorchat_split}-*.parquet")):
        parquet = pq.ParquetFile(path)
        columns = ["mode", "name", "processed_conversation", "textbook_folder"]
        for batch in parquet.iter_batches(batch_size=2000, columns=columns):
            for row in batch.to_pylist():
                for conversation in tutorchat_conversations(row, split):
                    kind = "exam" if row["name"] == "generateexam" else "dialogue"
                    yield f"tutorchat/{kind}", [conversation]


def prepare_split(split: str, source_names: list[str]):
    all_ids, all_mask, offsets, sources = [], [], [0], []
    stats = {}

    for source, conversations in iter_sources(split):
        if source not in source_names:
            source_names.append(source)
        source_stats = stats.setdefault(source, {"conversations": 0, "truncated": 0, "tokens": 0, "trained": 0})
        for ids, mask, truncated in encode(conversations):
            all_ids.append(np.array(ids, dtype=np.uint16))
            all_mask.append(np.array(mask, dtype=np.uint8))
            offsets.append(offsets[-1] + len(ids))
            sources.append(source_names.index(source))
            source_stats["conversations"] += 1
            source_stats["truncated"] += truncated
            source_stats["tokens"] += len(ids)
            source_stats["trained"] += sum(mask)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{split}_ids.npy", np.concatenate(all_ids))
    np.save(OUT_DIR / f"{split}_mask.npy", np.concatenate(all_mask))
    np.save(OUT_DIR / f"{split}_offsets.npy", np.array(offsets, dtype=np.int64))
    np.save(OUT_DIR / f"{split}_sources.npy", np.array(sources, dtype=np.uint8))

    print(f"\n{split}:")
    for source, s in stats.items():
        print(
            f"  {source:40} {s['conversations']:>8,} conversations "
            f"({s['truncated']:,} truncated), {s['tokens']:>12,} tokens, {s['trained']:>12,} trained"
        )
    total = sum(s["tokens"] for s in stats.values())
    print(f"  total: {len(sources):,} conversations, {total:,} tokens")


if __name__ == "__main__":
    source_names = []
    prepare_split("train", source_names)
    prepare_split("val", source_names)
    with open(OUT_DIR / "sources.json", "w") as f:
        json.dump(source_names, f, indent=2)

    # Check one conversation
    ids = np.load(OUT_DIR / "val_ids.npy", mmap_mode="r")
    mask = np.load(OUT_DIR / "val_mask.npy", mmap_mode="r")
    offsets = np.load(OUT_DIR / "val_offsets.npy")
    start, end = offsets[0], offsets[1]
    print("\nFirst val conversation:")
    print(tok.decode(ids[start:end].tolist(), skip_special_tokens=False))
    print("\nTrained part:")
    print(tok.decode([t for t, m in zip(ids[start:end], mask[start:end]) if m], skip_special_tokens=False))
