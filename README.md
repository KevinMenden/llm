# Todo

- add MoE
- better dataset
- add dropout
- grouped query attention
- bigger vocabulary

### Before the big training run

Needed for chat, SFT and RLHF:

- `generate`: stop on an end token, add top-k/top-p, remove the unused `n_tokens`
- train with a longer `block_size` (512+), 256 tokens is short for a conversation
- SFT batching: right-pad and set labels to -100 for padding and user turns (`cross_entropy` already ignores -100, and right padding needs no attention mask because of the causal mask)
- `forward`: option to return hidden states, for a value or reward head (only needed for PPO, not for DPO)

Training pipeline:

- `torch.compile` for training (keep the uncompiled model for `generate` and checkpoints)
- gradient checkpointing, if 100M params don't fit at the wanted batch size

Architecture (nice to have):

- KV cache, worth it once RLHF needs many rollouts


Probably skip for now:

- dropout, 2B+ tokens on 100M params won't overfit (maybe useful for SFT)
- grouped query attention, mostly useful together with a KV cache
- MoE, too much complexity at this model size


### Notes

Hygiene. Add gradient clipping at 1.0, weight decay on the matrices only, weight tying between the embedding and output head, and saving checkpoints. A checkpoint save is cheap and means you won’t lose a long run.
