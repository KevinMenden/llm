# Todo

- add RoPE
- add MoE
- better dataset
- better training tracking
- add dropout


### Notes
How long to train
The usual rule of thumb (Chinchilla) is about 20 tokens per parameter. That’s roughly 30M tokens for a model your size, and you pass that in the first few hundred steps.
So you’re capacity-limited, not data-limited. Training for 10k steps will keep improving, but the curve flattens quickly, and more steps won’t turn this model into a good one.
Run it until the val loss stops dropping. Watch for the train/val gap. With 178M tokens and 1.4M parameters, I wouldn’t expect much overfitting.
Where the loss should end up
This is a guess, since I don’t know your tokenizer exactly. Random init is about 7.6 and a unigram baseline is roughly 6. I’d expect a model this size to settle somewhere around 4.0 to 4.5 on WikiText-103, a perplexity of about 55 to 90.
At that level, the samples should be grammatical-looking, Wikipedia-flavored fragments. Expect them to drift off topic and make up facts after a sentence or two. That’s normal for 1.4M parameters.
Loss values depend on the tokenizer. A 2000-token vocab means each token carries less information than in a 50k vocab, so don’t compare your numbers with GPT-2 figures.
Good next steps, roughly in order of payoff

Learning-rate schedule. Your lr is a constant 1e-3. Add a short warmup (about 100 steps) followed by cosine decay. This is usually the cheapest improvement, and the decay phase often gives a visible drop in loss at the end.

Make the model bigger. Try n_dim=384, 6 layers and 6 heads, which is about 15M parameters. This is the biggest lever, because you’re capacity-limited. Check that you still fit in GPU memory at batch 256.

Bigger vocabulary. Retrain the tokenizer at 8k to 16k. Token sequences get shorter, so each context window covers more text. The output layer grows, so do this together with the bigger model.


Hygiene. Add gradient clipping at 1.0, weight decay on the matrices only, weight tying between the embedding and output head, and saving checkpoints. A checkpoint save is cheap and means you won’t lose a long run.
Tracking. Log train and val loss to a file or to TensorBoard or Weights & Biases. This is on your README list, and comparing runs gets hard without it.
Your README list
RoPE is a good next architectural change. It replaces the learned position embeddings and usually helps, and it’s a nice exercise.
MoE is more interesting once the model is larger and the basics are tuned. At this size it just adds complexity.
Dropout I’d skip for now. With 178M tokens and a small model you’re underfitting, and dropout only helps when you overfit.
I’d do the LR schedule first and then the bigger model, so you can see how much each one changes the loss.
