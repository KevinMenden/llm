# Todo

- add MoE
- better dataset
- add dropout
- grouped query attention
- bigger vocabulary


### Notes

Hygiene. Add gradient clipping at 1.0, weight decay on the matrices only, weight tying between the embedding and output head, and saving checkpoints. A checkpoint save is cheap and means you won’t lose a long run.
