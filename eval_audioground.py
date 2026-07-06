#!/usr/bin/env python3
"""
Evaluate AudioGround (SALMONN-based) on CM_test or UnAV-100.

Single-GPU:
    CUDA_VISIBLE_DEVICES=4 python eval_audioground.py \\
        --dataset cm_test \\
        --ckpt output/audioground/checkpoint_best.pth

Multi-GPU (4 GPUs):
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 eval_audioground.py \\
        --dataset cm_test \\
        --ckpt output/audioground/checkpoint_best.pth

Outputs:
    <output_dir>/<dataset>_<tag>_preds.json   -- per-sample predictions
    <output_dir>/<dataset>_<tag>_metrics.json -- R1@0.3/0.5/0.7 + mIoU
"""

import argparse
import json
import logging
import os
import re
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import WhisperFeatureExtractor
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from models.audioground import AudioGround

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Dataset paths ──────────────────────────────────────────────────────────────
DATASETS = {
    "cm_test": {
        "ann": "/home/irteam/Qwen2Audio-GRPO/_DATASETS/Clotho-Moment/test/test.json",
        "dur_key": "total_duration",
        "id_key":  "recipe_id",
    },
    "unav100": {
        "ann": "/home/irteam/Qwen2Audio-GRPO/_DATASETS/unav100-subset/unav100-subset_reformatted.json",
        "dur_key": "duration",
        "id_key":  "qid",
    },
    "tut2017": {
        "ann": "/home/irteam/Qwen2Audio-GRPO/_DATASETS/tutse2017/tut2017_eval.json",
        "dur_key": "duration",
        "id_key":  "qid",
    },
}

# ── Model config (matches training) ───────────────────────────────────────────
MODEL_CFG = dict(
    llama_path  = "/dev/shm/audioground_models/salmonn-7b-llama-merged",
    whisper_path= "/dev/shm/audioground_models/whisper-large-v3",
    beats_path  = "/dev/shm/audioground_models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    ckpt        = "/home/irteam/avqa2/SALMONN/ckpts/salmonn_7b_v0.pth",
    freeze_whisper=True, freeze_beats=True,
    use_speech_Qformer=True, freeze_speech_QFormer=False,
    window_level_Qformer=True, num_speech_query_token=1,
    second_per_window=0.333333, second_stride=0.333333,
    speech_llama_proj_model="", freeze_speech_llama_proj=False,
    lora=True, lora_rank=32, lora_alpha=64, lora_dropout=0.1,
    multi_prompt=True, prompt_path="prompts/train_prompt.json",
    prompt_template="USER: {}\nASSISTANT:",
    max_txt_len=300, end_sym="</s>",
    low_resource=False, device_8bit=0,
    use_frame_interpolation=True,
    use_timestamp_conditioning=True,
    use_absolute_time_embedding=True,
    ate_type="hybrid", ate_beta_init=0.05,
)

MAX_NEW_TOKENS  = 100
PROMPT_TEMPLATE = "USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:"
GENERATE_CFG = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "num_beams": 4,
    "do_sample": False,
    "min_length": 1,
    "temperature": 1.0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
}


# ── Evaluation dataset ─────────────────────────────────────────────────────────

class EvalDataset(Dataset):
    CHUNK_SEC = 30
    SR        = 16_000

    def __init__(self, ann_path, whisper_path, dur_key, id_key):
        data = json.load(open(ann_path))
        self.ann = data["annotation"] if "annotation" in data else data
        self.wav_proc   = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.chunk_samp = self.CHUNK_SEC * self.SR
        self.dur_key    = dur_key
        self.id_key     = id_key

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        item = self.ann[idx]

        audio, sr = sf.read(item["path"])
        if audio.ndim == 2:
            audio = audio[:, 0]

        if sr != self.SR:
            new_len = int(len(audio) * self.SR / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)), audio,
            ).astype(np.float32)

        audio = audio.astype(np.float32)
        if len(audio) < self.SR:
            audio = np.pad(audio, (0, self.SR - len(audio)))

        # dynamic chunking: ceil(audio_len / chunk_samp) chunks
        import math
        n_chunks = max(1, math.ceil(len(audio) / self.chunk_samp))
        padded = np.zeros(n_chunks * self.chunk_samp, dtype=np.float32)
        padded[:len(audio)] = audio

        specs = []
        for i in range(n_chunks):
            chunk = padded[i * self.chunk_samp: (i + 1) * self.chunk_samp]
            feat  = self.wav_proc(chunk, sampling_rate=self.SR, return_tensors="pt")
            specs.append(feat["input_features"].squeeze(0))
        spectrogram = torch.stack(specs, dim=0)

        return {
            "spectrogram": spectrogram,
            "raw_wav":     torch.from_numpy(audio),
            "Q":           item.get("Q", ""),
            "gt":          item.get("text", ""),
            "duration":    float(item.get(self.dur_key, 60.0)),
            "id":          str(item.get(self.id_key, item["path"])),
        }

    def collate(self, samples):
        # pad spectrograms to max n_chunks in batch
        max_chunks = max(s["spectrogram"].shape[0] for s in samples)
        spec_shape = samples[0]["spectrogram"].shape[1:]  # (mel_bins, time)
        padded_specs = []
        for s in samples:
            sp = s["spectrogram"]
            if sp.shape[0] < max_chunks:
                pad = torch.zeros(max_chunks - sp.shape[0], *spec_shape)
                sp = torch.cat([sp, pad], dim=0)
            padded_specs.append(sp)
        spectrogram = torch.stack(padded_specs)

        from torch.nn.utils.rnn import pad_sequence
        raw_wavs = [s["raw_wav"] for s in samples]
        lengths  = torch.tensor([r.shape[0] for r in raw_wavs])
        raw_wav  = pad_sequence(raw_wavs, batch_first=True, padding_value=0.0)
        padding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)

        return {
            "spectrogram":  spectrogram,
            "raw_wav":      raw_wav,
            "padding_mask": padding_mask,
            "Q":        [s["Q"]        for s in samples],
            "gt":       [s["gt"]       for s in samples],
            "duration": [s["duration"] for s in samples],
            "id":       [s["id"]       for s in samples],
        }


# ── Timestamp parsing ──────────────────────────────────────────────────────────

_TS_PATTERNS = [
    re.compile(r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)"),      # "X to Y"
    re.compile(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)"),         # "X - Y"
    re.compile(r"(\d+\.?\d*)\s+and\s+(\d+\.?\d*)"),       # "X and Y"
]

def parse_timestamps(text: str):
    """Return (start, end) floats or None if parse fails."""
    # strip answer tags if present
    text = re.sub(r"<[^>]+>", " ", text)
    for pat in _TS_PATTERNS:
        m = pat.search(text)
        if m:
            s, e = float(m.group(1)), float(m.group(2))
            if e < s:
                s, e = e, s
            return s, e
    return None


def parse_gt(gt_text: str):
    """Parse ground-truth timestamp string."""
    return parse_timestamps(gt_text)


# ── Metrics ────────────────────────────────────────────────────────────────────

def iou(pred, gt):
    """Intersection over Union for two (start, end) intervals."""
    inter = max(0.0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return inter / union if union > 0 else 0.0


def compute_metrics(records):
    thresholds = [0.3, 0.5, 0.7]
    ious, parse_ok = [], 0

    for r in records:
        pred = parse_timestamps(r["pred"])
        gt   = parse_gt(r["gt"])
        if pred is None or gt is None:
            r["iou"] = None
            r["parse_ok"] = False
            continue
        r["parse_ok"] = True
        parse_ok += 1
        v = iou(pred, gt)
        r["iou"] = v
        ious.append(v)

    n = len(records)
    metrics = {
        "n_samples":    n,
        "parse_rate":   parse_ok / n if n else 0,
        "mIoU":         float(np.mean(ious)) if ious else 0.0,
    }
    for thr in thresholds:
        metrics[f"R1@{thr}"] = float(np.mean([v >= thr for v in ious])) if ious else 0.0

    return metrics


# ── Distributed helpers ────────────────────────────────────────────────────────

def is_dist():
    return dist.is_available() and dist.is_initialized()


def gather_list(local_list):
    """Gather a list of dicts from all ranks to rank 0."""
    if not is_dist():
        return local_list
    world = dist.get_world_size()
    buf = [None] * world
    dist.all_gather_object(buf, local_list)
    return [item for sub in buf for item in sub]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--ckpt",    default="", help="training checkpoint (trainable params)")
    parser.add_argument("--batch",   type=int, default=8)
    parser.add_argument("--out",     default="output/eval")
    parser.add_argument("--tag",     default="")
    parser.add_argument("--no_temporal", action="store_true",
                        help="disable FI/TS/ATE (for ablation_no_fi checkpoint)")
    parser.add_argument("--fi_only", action="store_true",
                        help="enable FI only, disable TS/ATE (for ablation_fi_only checkpoint)")
    args = parser.parse_args()

    # ── Init distributed ──────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main    = local_rank == 0

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")

    # ── Load model ────────────────────────────────────────────────────────────
    if args.no_temporal:
        MODEL_CFG["use_frame_interpolation"]    = False
        MODEL_CFG["use_timestamp_conditioning"] = False
        MODEL_CFG["use_absolute_time_embedding"]= False
    elif args.fi_only:
        MODEL_CFG["use_frame_interpolation"]    = True
        MODEL_CFG["use_timestamp_conditioning"] = False
        MODEL_CFG["use_absolute_time_embedding"]= False

    if is_main:
        log.info("Loading AudioGround model...")
    model = AudioGround.from_config(MODEL_CFG)

    # Load training checkpoint (trainable params only)
    if args.ckpt and os.path.isfile(args.ckpt):
        if is_main:
            log.info(f"Loading training checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_main:
            log.info(f"  Loaded {len(state) - len(unexpected)} keys "
                     f"({len(missing)} missing, {len(unexpected)} unexpected)")
    elif is_main:
        log.warning("No training checkpoint provided — evaluating base pretrained model.")

    model = model.to(device).eval()

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    ds_cfg = DATASETS[args.dataset]
    dataset = EvalDataset(
        ann_path    = ds_cfg["ann"],
        whisper_path= MODEL_CFG["whisper_path"],
        dur_key     = ds_cfg["dur_key"],
        id_key      = ds_cfg["id_key"],
    )
    if is_main:
        log.info(f"Dataset: {args.dataset}  ({len(dataset)} samples)")

    sampler = DistributedSampler(dataset, shuffle=False) if world_size > 1 else None
    loader  = DataLoader(
        dataset,
        batch_size  = args.batch,
        sampler     = sampler,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = dataset.collate,
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    records = []
    for batch_idx, batch in enumerate(loader):
        samples = {
            "spectrogram":  batch["spectrogram"].to(device),
            "raw_wav":      batch["raw_wav"].to(device),
            "padding_mask": batch["padding_mask"].to(device),
        }
        prompts = [PROMPT_TEMPLATE.format(q) for q in batch["Q"]]

        with torch.no_grad():
            preds = model.generate(samples, GENERATE_CFG, prompts=prompts)

        # Strip prompt prefix from each output
        for i, (pred_full, gt, dur, sid) in enumerate(
            zip(preds, batch["gt"], batch["duration"], batch["id"])
        ):
            # model.generate returns full decoded text including the prompt
            pred = pred_full
            if "ASSISTANT:" in pred:
                pred = pred.split("ASSISTANT:")[-1].strip()
            # strip EOS token
            pred = pred.replace("</s>", "").strip()

            records.append({
                "id":   sid,
                "Q":    batch["Q"][i],
                "gt":   gt,
                "pred": pred,
                "duration": float(dur),
            })

        if is_main and (batch_idx + 1) % 20 == 0:
            log.info(f"  [{batch_idx+1}/{len(loader)}]  processed {len(records)} samples")

    # ── Gather from all ranks → rank 0 ────────────────────────────────────────
    all_records = gather_list(records)

    if is_main:
        # Deduplicate (DistributedSampler may pad last batch)
        seen, deduped = set(), []
        for r in all_records:
            if r["id"] not in seen:
                seen.add(r["id"])
                deduped.append(r)
        all_records = deduped

        metrics = compute_metrics(all_records)

        tag = f"_{args.tag}" if args.tag else ""
        ckpt_tag = os.path.splitext(os.path.basename(args.ckpt))[0] if args.ckpt else "base"
        stem = f"{args.dataset}_{ckpt_tag}{tag}"

        os.makedirs(args.out, exist_ok=True)
        preds_path   = os.path.join(args.out, f"{stem}_preds.json")
        metrics_path = os.path.join(args.out, f"{stem}_metrics.json")

        with open(preds_path, "w") as f:
            json.dump(all_records, f, indent=2, ensure_ascii=False)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        log.info(f"\n{'='*50}")
        log.info(f"Dataset : {args.dataset}  ({metrics['n_samples']} samples)")
        log.info(f"Ckpt    : {args.ckpt or 'base pretrained'}")
        log.info(f"Parse   : {metrics['parse_rate']*100:.1f}%")
        log.info(f"mIoU    : {metrics['mIoU']*100:.2f}")
        log.info(f"R1@0.3  : {metrics['R1@0.3']*100:.2f}")
        log.info(f"R1@0.5  : {metrics['R1@0.5']*100:.2f}")
        log.info(f"R1@0.7  : {metrics['R1@0.7']*100:.2f}")
        log.info(f"Preds   : {preds_path}")
        log.info(f"Metrics : {metrics_path}")
        log.info("="*50)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
