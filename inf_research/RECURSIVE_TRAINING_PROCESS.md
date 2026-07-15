# Recursive Diffushadow Training Process

This project uses a recursive training process to extend Diffushadow from
small, exactly generated systems to larger qubit sizes.

## Core Idea

The training data for later Diffushadow models is not produced only by exact
calculation. Instead, each trained model can generate larger-size data, and
that model-generated data is then mixed with trusted exact data to train the
next model.

In this setup, the model index and the qubit size are coupled: an `.npz`
evaluation file at a larger qubit size may come from a later recursive model,
not from the same model evaluated at a different system size.

## Example Schedule

Round 1:

- Exact data: `N = 6, 8, 10, 12`
- Training script: `train_oseq.py`
- Output model: `Diffushadow1`

Round 2:

- Exact data kept in the training mix: `N = 10, 12`
- Model-generated data from `Diffushadow1`: `N = 18, 20, 22, 24`
- Training script: `train_oseq.py`
- Output model: `Diffushadow2`

Later rounds:

- Keep a small set of exact or trusted anchor sizes.
- Use the previous Diffushadow model to generate data at larger qubit
  sizes.
- Train the next Diffushadow model on the mixed exact-plus-generated dataset.

Symbolically:

```text
Exact small-N data
    -> train Diffushadow1
    -> generate larger-N data
    -> mix exact anchors + generated data
    -> train Diffushadow2
    -> generate still larger-N data
    -> ...
```
