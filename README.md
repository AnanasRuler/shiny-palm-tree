# SF-Fuse

A deep learning framework for genomic function prediction using MLM (Masked Language Model) pre-training and parameter-efficient fine-tuning.

## Overview

This project implements a two-stage training strategy for genomic sequence modeling:

1. **MLM Pre-training** — Encoder-only masked language modeling on reference genome sequences
2. **Fine-tuning** — Task-specific training on genomic functional tracks (5,313 tracks from Basenji2)

Supports multiple encoder architectures:
- **Caduceus** — Mamba-CNN-Transformer hybrid encoder
- **HyenaDNA** — Long-range genomic language model (via HuggingFace)
- **Wisteria** — 12-layer transformer encoder

## Quick Start

### Setup

```bash
conda create -n sf-fuse python=3.12
conda activate sf-fuse
pip install -r requirements.txt
# mamba-ssm and causal-conv1d require local CUDA-compatible wheels
pip install mamba-ssm causal-conv1d
```

### Training

```bash
# Fine-tuning (default: Caduceus encoder)
python train.py

# MLM pre-training (encoder-only)
python pretrain_mlm.py

# HyenaDNA fine-tuning
python train.py task=dualrep_hyenadna
```

### Evaluation

```bash
python test.py --ckpt_path <checkpoint.ckpt> --test_data <test.h5> --output_dir <dir>
```

## Configuration

Uses Hydra for configuration management. Key override syntax:

```bash
python train.py task.n_pre_layers=0  # CNN Stem
python train.py task.n_pre_layers=8  # Encoder-only
python train.py task.d_model=512     # Change model dimension
python train.py task.use_lora=true   # Enable LoRA
```

See `configs/` for all configuration files.

## Data Format

- **Fine-tuning data:** H5 files with `sequences` (One-Hot bool, B×L×4) and `targets` (float32, B×896×T)
- **MLM pre-training data:** FASTA + BED files
- **Sequence length:** 131,072 bp (SF-Fuse standard)

Place data files in `data/` directory.

## Architecture

```
DNA Sequence (B, 131072) → Encoder → CNN Bridge (128× downsampling) → Decoder → Head (896×5313)
```

- **Encoder:** 8-layer Mamba-CNN-Transformer hybrid (GCMB + BiMamba + FoPE)
- **CNN Bridge:** 7-stage downsampling (131k → 1k)
- **Decoder:** GatedAttentionDecoder (11 layers)
- **Head:** Center crop (896 bins) + MLP (5,313 tracks)

## Hardware Requirements

- GPU with ≥48GB VRAM recommended (tested on NVIDIA RTX PRO 6000 Blackwell, 96GB)
- Multi-GPU: Set `trainer.devices=N` for DDP

## Token IDs

CaduceusTokenizer: `[CLS]=0, [SEP]=1, [BOS]=2, [MASK]=3, [PAD]=4, [RESERVED]=5, [UNK]=6, A=7, C=8, G=9, T=10, N=11`

## License

MIT
