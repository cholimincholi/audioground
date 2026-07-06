# AudioGround: Fine-Grained Temporal Grounding in Audio via Deterministic Boundary Supervision

AudioGround equips Large Audio Language Models (LALMs) with **fine-grained temporal grounding** — not just recognizing *which* sounds occur, but localizing *when* they occur with explicit start–end timestamps `[ts, te]`.

It consists of two parts:

1. **AudioGround-IT** — a time-aware audio instruction-tuning dataset with **deterministic boundary supervision**: ~49.9K instructions over ~835 hours of audio across four temporal tasks (Grounding, Duration, Ordering, Frequency), built by concatenating AudioCaps clips so that every event boundary is known by construction (no post-hoc or LLM-inferred labels).
2. **AudioGround** — a lightweight temporal extension of [SALMONN](https://github.com/bytedance/SALMONN) that adds (i) frame-level interpolation across the Whisper and BEATs encoders, (ii) a **time-aware Q-Former** with sliding-window compression and timestamp conditioning, and (iii) a **hybrid absolute time embedding** (sinusoidal base + zero-initialized learnable residual).

> This repository builds on ByteDance/Tsinghua's SALMONN. The base SALMONN code is retained; AudioGround adds the dataset-synthesis pipeline, the temporal-awareness model components, and grounding evaluation. See the upstream project for the original model.

## Method overview

**Data synthesis (AudioGround-IT).** Each AudioCaps clip (~10 s) is an atomic building block. Clips are concatenated with short silence gaps (0.3–0.5 s) into 20–120 s audio. Because the concatenation order is controlled, every event boundary is deterministic. Semantic ambiguity is controlled with a caption cosine-similarity threshold (τ = 0.33), and both temporal position and length are balanced to reduce distributional bias. All four tasks share a unified output target: the temporal interval(s) `[ts, te]` supporting the answer.

**Model (AudioGround).** Long audio is split into 30 s chunks and encoded by frozen Whisper + BEATs encoders (BEATs frames interpolated to match Whisper). A sliding-window Q-Former compresses each window and is conditioned on a textual timestamp string ("This segment is from s to e seconds"). A hybrid absolute time embedding is added to each window's tokens to inject continuous global time. Only the window-level Q-Former, the projection layer, and the LoRA adapters on the frozen LLM are trained.

## Repository layout

| Path | Purpose |
|---|---|
| `generate_audiocaps_paper.py` | Synthesize 20–120 s audio by concatenating AudioCaps clips (deterministic boundaries) |
| `build_audioground_it.py` | Build the 4-task AudioGround-IT instruction JSON |
| `dataset_audioground.py` | Instruction-tuning dataset loader with long-audio chunking |
| `models/salmonn.py` | Base SALMONN + sliding-window Q-Former + time-embedding modules |
| `models/audioground.py` | AudioGround model (absolute time embedding: sinusoidal / learned / hybrid) |
| `train.py` | Training entry point (config-driven) |
| `eval_audioground.py` | Moment-retrieval evaluation (R1@IoU) |
| `configs/audioground.yaml` | Main training config (paper settings) |
| `소프트웨어_등록_AudioGround/` | Software-registration package (document + core code copies) |

## Setup

```bash
pip install -r requirements.txt
```

Download the frozen backbones and place them where `configs/audioground.yaml` expects:
- [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3)
- [Fine-tuned BEATs_iter3+ (AS2M) cpt2](https://github.com/microsoft/unilm/tree/master/beats)
- A merged SALMONN-7B LLM base (see `merge_salmonn_lora.py`) and the SALMONN checkpoint `salmonn_7b_v0.pth`

## Usage

### 1. Build AudioGround-IT

```bash
# Synthesize long audio with deterministic event boundaries
python generate_audiocaps_paper.py \
    --source_dir /path/to/AudioCaps \
    --output_dir data/AudioCapsConcat_paper \
    --train_samples 15000 \
    --mode_ratio long:0.55,single:0.35,multi:0.10 \
    --workers 16

# Assemble the 4-task instruction JSON
python build_audioground_it.py
```

### 2. Train

```bash
# Paper settings: 6K steps, effective batch 24, LoRA r=32 (α=64)
bash train_audioground.sh
# or directly:
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path configs/audioground.yaml
```

Key config knobs (`configs/audioground.yaml`): `second_per_window` (sliding window), `use_timestamp_conditioning` (TS), `use_absolute_time_embedding` / `ate_type` (`sinusoidal` | `learned` | `hybrid`), `ate_beta_init`.

### 3. Evaluate (moment retrieval, R1@IoU)

```bash
bash run_eval_audioground.sh output/audioground/checkpoint_best.pth all
# or a single benchmark:
python eval_audioground.py --dataset cm_test --ckpt output/audioground/checkpoint_best.pth
```

Benchmarks: Clotho-Moment, UnAV-100-subset, TUT-Sound-Events-2017, reported as R1@0.5 / R1@0.7.

## Acknowledgements

Built on [SALMONN](https://github.com/bytedance/SALMONN) (Tsinghua University & ByteDance). Audio sourced from [AudioCaps](https://audiocaps.github.io/). We thank the authors of these projects.
