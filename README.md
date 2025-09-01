# RxNet

Character-level MLP that learns and predicts pharmaceutical OTC/prescription brand names.

## Approach and inspiration

This project implements the core idea from Bengio et al. (2003) — a neural
probabilistic language model operating over a fixed-size context — adapted to
next-character prediction. Instead of word tokens, we use character tokens with
start-of-sequence `^` and end-of-sequence `$` markers. A small embedding maps
each character id to a continuous vector, the context window is flattened, and
then a multilayer perceptron produces logits over the vocabulary for the next
character.

The project was also inspired by Freeman Jiang’s Japanese city name generator, which follows the same MLP
language-modeling design. See
[Freeman Jiang’s jp-city-generator](https://github.com/freeman-jiang/jp-city-generator).

## Model architecture

- **Embedding**: `Embedding(vocab_size, 32)`
- **Context window**: length `CONTEXT` (default 7)
- **Feed-forward MLP**:
  - `Linear(CONTEXT * 32 → 20 * CONTEXT)` → ReLU → Dropout(p=0.2)
  - `Linear(20 * CONTEXT → 15 * CONTEXT)` → ReLU → Dropout(p=0.2)
  - `Linear(15 * CONTEXT → vocab_size)`

## Data and preprocessing

- Input file: `rxnorm-names.csv` with a column `STR` containing brand names.
- Strings are trimmed and normalized to Unicode NFC.
- Vocabulary includes all observed characters plus special tokens: `^`, `$`.
- Training pairs are created by sliding a fixed left context over each name and
  predicting the next character; a final pair predicts `$` after the last char.

**Dataset**: `rxnorm-names.csv` is a curated and cleaned list of
brand names derived from the RxNorm dataset maintained by the U.S. National
Library of Medicine. This repository only includes the names used for
modeling; additional curation steps were performed offline.

## Training

All training logic is in `train.py`:

- Optimizer: AdamW (`lr=1e-4`, `weight_decay=1e-5`)
- Scheduler: ReduceLROnPlateau (`factor=0.1`, `patience=3`)
- Loss: CrossEntropyLoss with label smoothing (`0.05`)
- Mixed precision: `torch.cuda.amp` (when CUDA is available)
- EMA: Exponential moving average of weights (`decay=0.999`) for evaluation
- Early stopping: patience of 7 epochs without validation loss improvement

## Quick start

1. Install dependencies (PyTorch, pandas, scikit-learn, tqdm, torch-ema).
3. Train:

```bash
python train.py
```