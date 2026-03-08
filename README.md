# Single-Pass Evidence Extraction via Hidden State Classification

Classify which passages are relevant to a multi-hop question by extracting hidden states from a single LLM forward pass — no autoregressive decoding required.

## Method

Given a query and N candidate passages, we concatenate them into a single input sequence and run one forward pass through a frozen LLM. The hidden state at each passage's EOS token is fed into a linear classification head to predict relevance (0/1). This enables inter-passage interaction while avoiding the cost of autoregressive generation.

We compare two backbone variants:
- **Causal LM** (Qwen3-0.6B): unidirectional inter-passage interaction
- **Bidirectional** (Qwen3-Embedding-0.6B): full bidirectional inter-passage interaction

Against two baselines:
- **Dual-Tower** (all-MiniLM-L6-v2): cosine similarity, no interaction
- **Cross-Encoder** (ms-marco-MiniLM-L6-v2): query-passage interaction only

## Project Structure

```
├── experiment.ipynb        # End-to-end experiment notebook (run on Colab)
├── src/
│   ├── config.py           # Experiment configuration
│   ├── data.py             # HotpotQA data loading and preprocessing
│   ├── modeling.py         # Backbone loading, tokenization, classifier model
│   ├── train_eval.py       # Training loop, evaluation, threshold tuning
│   └── utils.py            # Helpers (seed, device, JSON I/O, formatting)
├── scripts/
│   └── draw_architecture.py # Architecture diagram generation
├── assets/                  # Generated figures (architecture diagrams)
├── proposal/
│   └── proposal.md         # Research proposal
├── tests/                  # Unit tests for src/
└── artifacts/runs/         # Saved results (generated after running experiments)
    ├── causal/
    ├── bidirectional/
    ├── dual_tower/
    └── cross_encoder/
```

## Requirements

```
torch
transformers
datasets
sentence-transformers
tqdm
matplotlib
```

## Usage

1. Zip the `src/` folder and upload to Google Colab
2. Upload `experiment.ipynb` to Colab
3. Run all cells sequentially

The notebook handles data downloading, model loading, training, evaluation, and visualization. Results are saved to `artifacts/runs/`.

## Results (HotpotQA Distractor, Paragraph-level F1)

| Method | Overall F1 | Bridge F1 | Comparison F1 |
|--------|-----------|-----------|---------------|
| Dual-Tower | 0.519 | 0.496 | 0.616 |
| Cross-Encoder | 0.578 | 0.567 | 0.630 |
| Bidirectional | 0.604 | 0.588 | 0.667 |
| **Causal LM** | **0.675** | **0.659** | **0.737** |

The causal LM with a frozen backbone and a single linear head achieves the best performance, outperforming both baselines and the bidirectional variant.
