# Todo

- add RoPE
- add MoE
- better dataset
- add dropout


### Notes

Bigger vocabulary. Retrain the tokenizer at 8k to 16k. Token sequences get shorter, so each context window covers more text. The output layer grows, so do this together with the bigger model.
Hygiene. Add gradient clipping at 1.0, weight decay on the matrices only, weight tying between the embedding and output head, and saving checkpoints. A checkpoint save is cheap and means you won’t lose a long run.


