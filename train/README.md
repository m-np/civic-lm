# train/

SFT with LoRA (Nov 3–9). Reads only the `train` split of the bench release. Training items come
from the same generators, but from CFR parts that are disjoint from dev and test parts (see
`configs/scope.yaml`). DPO only if time remains. No continued pretraining.
