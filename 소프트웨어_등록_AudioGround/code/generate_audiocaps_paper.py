#!/usr/bin/env python3
"""
Generate AudioCapsConcat dataset (paper spec: 20-120s audio, long-mode events 15-40s).

Usage:
    python generate_audiocaps_paper.py \
        --source_dir /home/irteam/Qwen2Audio-GRPO/_DATASETS/AudioCaps \
        --output_dir /home/irteam/avqa2/SALMONN/data/AudioCapsConcat_paper \
        --train_samples 15000 \
        --mode_ratio long:0.55,single:0.35,multi:0.10 \
        --workers 16
"""

import argparse
import json
import os
import multiprocessing as mp
from pathlib import Path
from collections import namedtuple, Counter, OrderedDict
from functools import partial

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal

# ──────────────────────────── Config (paper spec: 20-120s) ──────

MIN_CLIPS = 3             # minimum 3 clips for 20s+ audio
MAX_CLIPS = 20            # up to 20 clips for 120s audio
MIN_CROP_SEC = 2.0
MIN_TOTAL_SEC = 20.0      # paper: 20s minimum
MAX_TOTAL_SEC = 120.0     # paper: 120s maximum
SILENCE_MIN_SEC = 0.3
SILENCE_MAX_SEC = 5.0
SILENCE_PROB = 1.0
TARGET_SR = 16000
SOURCE_SR = 24000
BG_NOISE_AMP = 0.003
FADE_MS = 15

# Long mode: merge 2-4 adjacent clips (no gap) → 15-40s merged event
LONG_MERGE_MIN = 2
LONG_MERGE_MAX = 4
LONG_MIN_SEC = 15.0
LONG_MAX_SEC = 40.0
LONG_FEASIBLE_TOTAL = 20.0

# Multi mode config
MULTI_REPEAT_MIN = 2
MULTI_REPEAT_MAX = 3
MULTI_FEASIBLE_CLIPS = 3

SegmentInfo = namedtuple(
    "SegmentInfo",
    ["caption", "crop_start", "crop_duration", "concat_start", "concat_end", "gap_before", "long_group_id"],
    defaults=[None],
)

# ──────────────────────────── Gap / Fade helpers ───────────────

def make_gap_noise(n_samples: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate low-level background noise for gap sections (pink-ish noise)."""
    white = rng.randn(n_samples).astype(np.float32)
    # Simple 1-pole lowpass to approximate pink noise
    pink = np.empty_like(white)
    pink[0] = white[0]
    for i in range(1, n_samples):
        pink[i] = 0.7 * pink[i - 1] + 0.3 * white[i]
    return pink * BG_NOISE_AMP


def apply_fade(audio: np.ndarray, fade_in: bool = True, fade_out: bool = True) -> np.ndarray:
    """Apply short fade-in/out to avoid click artifacts at clip boundaries."""
    fade_samples = int(FADE_MS * TARGET_SR / 1000)
    if fade_samples < 1 or len(audio) < fade_samples * 2:
        return audio
    audio = audio.copy()
    if fade_in:
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples, dtype=np.float32)
    if fade_out:
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples, dtype=np.float32)
    return audio


# ──────────────────────────── Audio I/O ─────────────────────────

def load_and_preprocess(path: str) -> np.ndarray:
    """Load WAV, convert to mono float32, resample 24kHz→16kHz."""
    try:
        sr, data = wavfile.read(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")

    # to float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float64:
        data = data.astype(np.float32)

    # stereo → mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # resample if needed
    if sr != TARGET_SR:
        # use rational resampling
        from math import gcd
        g = gcd(TARGET_SR, sr)
        up, down = TARGET_SR // g, sr // g
        data = scipy.signal.resample_poly(data, up, down).astype(np.float32)

    return data


def save_wav(path: str, audio: np.ndarray, sr: int = TARGET_SR):
    """Save float32 audio as int16 WAV."""
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    wavfile.write(path, sr, audio_int16)


# ──────────────────────────── Sample Generation ─────────────────

def generate_single_sample(
    source_pool: list,
    rng: np.random.RandomState,
    force_duplicates: bool = False,
    target_duration: float = None,
) -> tuple:
    """
    Generate one concatenated sample with variable target duration.

    Args:
        force_duplicates: If True, intentionally repeat some clips for frequency task.
        target_duration: If None, randomly chosen from [MIN_TOTAL_SEC, MAX_TOTAL_SEC].

    Returns:
        audio: np.ndarray (float32, mono, TARGET_SR)
        segments: list of SegmentInfo
    """
    # Variable target duration: 매 sample마다 랜덤
    if target_duration is None:
        target_duration = rng.uniform(MIN_TOTAL_SEC, MAX_TOTAL_SEC)

    # target에 비례한 n_clips (avg 6초/클립 가정, ±2 변동)
    expected_clips = target_duration / 6.0
    n_clips = int(rng.uniform(expected_clips - 2.0, expected_clips + 2.0))
    # 안전 범위로 clip
    n_clips = max(MIN_CLIPS, min(n_clips, MAX_CLIPS))
    # target_duration / MIN_CROP_SEC 보다 클 수 없음 (수학적 제약)
    max_n_clips = max(MIN_CLIPS, int(target_duration / MIN_CROP_SEC))
    n_clips = min(n_clips, max_n_clips)

    # 이 샘플의 local min/max (target ± 더 넓은 buffer)
    local_min_total = max(target_duration - 5.0, MIN_TOTAL_SEC)
    local_max_total = min(target_duration + 5.0, MAX_TOTAL_SEC)

    # We'll iteratively build the sample, adjusting to hit 30-60s
    max_attempts = 20
    for attempt in range(max_attempts):
        if force_duplicates:
            # Pick fewer unique clips, then repeat some
            n_unique = rng.randint(2, max(3, n_clips - 1))  # 2 ~ n_clips-2 unique
            unique_indices = rng.randint(0, len(source_pool), size=n_unique)
            # Fill remaining slots by repeating from unique set
            n_extra = n_clips - n_unique
            extra_indices = unique_indices[rng.randint(0, n_unique, size=n_extra)]
            indices = np.concatenate([unique_indices, extra_indices])
            rng.shuffle(indices)  # randomize order
        else:
            indices = rng.randint(0, len(source_pool), size=n_clips)
        clips_info = []

        for idx in indices:
            src = source_pool[idx]
            audio_len = src["audio_len"]
            if audio_len < MIN_CROP_SEC:
                crop_dur = audio_len
                crop_start = 0.0
            else:
                crop_dur = rng.uniform(MIN_CROP_SEC, audio_len)
                crop_start = rng.uniform(0, audio_len - crop_dur)
            clips_info.append((idx, crop_start, crop_dur))

        # transitions
        gaps = [0.0]  # no gap before first clip
        for _ in range(n_clips - 1):
            if rng.random() < SILENCE_PROB:
                gaps.append(rng.uniform(SILENCE_MIN_SEC, SILENCE_MAX_SEC))
            else:
                gaps.append(0.0)

        total = sum(c[2] for c in clips_info) + sum(gaps)

        # Adjust if out of range (use local target-based bounds)
        if total > local_max_total:
            # shrink last clip
            excess = total - local_max_total
            last_dur = clips_info[-1][2]
            if last_dur - excess >= MIN_CROP_SEC:
                idx_l, cs_l, _ = clips_info[-1]
                clips_info[-1] = (idx_l, cs_l, last_dur - excess)
                total = local_max_total
            elif n_clips > MIN_CLIPS:
                # remove last clip and retry adjustment
                clips_info.pop()
                gaps.pop()
                n_clips -= 1
                total = sum(c[2] for c in clips_info) + sum(gaps)
                if total > local_max_total:
                    continue  # retry
            else:
                continue  # retry with new random

        if total < local_min_total:
            # try extending crop durations
            deficit = local_min_total - total
            for i in range(len(clips_info)):
                idx_c, cs_c, cd_c = clips_info[i]
                max_dur = source_pool[idx_c]["audio_len"]
                # extend towards end
                available = max_dur - (cs_c + cd_c)
                extend = min(available, deficit)
                if extend > 0:
                    clips_info[i] = (idx_c, cs_c, cd_c + extend)
                    deficit -= extend
                    if deficit <= 0:
                        break
            # extend towards start too
            if deficit > 0:
                for i in range(len(clips_info)):
                    idx_c, cs_c, cd_c = clips_info[i]
                    available = cs_c  # can extend start earlier
                    extend = min(available, deficit)
                    if extend > 0:
                        clips_info[i] = (idx_c, cs_c - extend, cd_c + extend)
                        deficit -= extend
                        if deficit <= 0:
                            break
            # if still short, add more clips
            if deficit > 0 and len(clips_info) < max_n_clips:
                while deficit > 0 and len(clips_info) < max_n_clips:
                    new_idx = rng.randint(0, len(source_pool))
                    src = source_pool[new_idx]
                    crop_dur = min(rng.uniform(MIN_CROP_SEC, src["audio_len"]), deficit + 1)
                    crop_dur = max(crop_dur, MIN_CROP_SEC)
                    crop_start = rng.uniform(0, max(0, src["audio_len"] - crop_dur))
                    clips_info.append((new_idx, crop_start, crop_dur))
                    if rng.random() < SILENCE_PROB:
                        g = rng.uniform(SILENCE_MIN_SEC, SILENCE_MAX_SEC)
                    else:
                        g = 0.0
                    gaps.append(g)
                    deficit -= (crop_dur + g)
            # if still short, retry with new random selection (no silence padding)
            if deficit > 0:
                continue

            total = sum(c[2] for c in clips_info) + sum(gaps)

        if local_min_total <= total <= local_max_total + 1.0:
            break
    else:
        # fallback: accept whatever we have
        pass

    # ── Build audio and segment info ──
    audio_parts = []
    segments = []
    current_pos = 0.0

    for i, (src_idx, crop_start, crop_dur) in enumerate(clips_info):
        src = source_pool[src_idx]
        gap = gaps[i] if i < len(gaps) else 0.0

        # add silence gap
        if gap > 0:
            silence_samples = int(round(gap * TARGET_SR))
            audio_parts.append(make_gap_noise(silence_samples, rng))
            current_pos += gap

        # load and crop audio
        audio_data = load_and_preprocess(src["audio_path"])
        start_sample = int(round(crop_start * TARGET_SR))
        end_sample = start_sample + int(round(crop_dur * TARGET_SR))
        end_sample = min(end_sample, len(audio_data))
        cropped = audio_data[start_sample:end_sample]

        actual_dur = len(cropped) / TARGET_SR
        concat_start = current_pos
        concat_end = current_pos + actual_dur

        audio_parts.append(apply_fade(cropped))
        segments.append(
            SegmentInfo(
                caption=src["caption"],
                crop_start=round(crop_start, 3),
                crop_duration=round(actual_dur, 3),
                concat_start=round(concat_start, 3),
                concat_end=round(concat_end, 3),
                gap_before=round(gap, 3),
            )
        )
        current_pos = concat_end

    # handle any trailing silence from deficit padding
    if len(gaps) > len(clips_info):
        trailing = gaps[-1]
        if trailing > 0:
            audio_parts.append(make_gap_noise(int(round(trailing * TARGET_SR)), rng))

    audio = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)

    # hard truncate to MAX_TOTAL_SEC
    max_samples = int(round(MAX_TOTAL_SEC * TARGET_SR))
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        # adjust last segment if needed
        if segments and segments[-1].concat_end > MAX_TOTAL_SEC:
            s = segments[-1]
            segments[-1] = s._replace(concat_end=round(MAX_TOTAL_SEC, 3))

    return audio, segments


def generate_long_sample(
    source_pool: list,
    rng: np.random.RandomState,
) -> tuple:
    """
    Generate a sample with one long merged segment (15-30s, gap-free).

    Structure: [filler...] [LONG: clip_a + clip_b + clip_c (no gap)] [filler...]
    The merged segment is recorded as individual segments in metadata but with
    gap_before=0 between merged clips, plus a 'long_group' field.

    Returns:
        audio: np.ndarray, segments: list of SegmentInfo
    """
    n_merge = rng.randint(LONG_MERGE_MIN, LONG_MERGE_MAX + 1)  # 2 or 3

    max_attempts = 30
    for _ in range(max_attempts):
        # Pick clips for the long segment
        merge_indices = rng.randint(0, len(source_pool), size=n_merge)
        merge_clips = []
        for idx in merge_indices:
            src = source_pool[idx]
            audio_len = src["audio_len"]
            # Use longer crops for long segments (5s ~ full length)
            min_crop = min(5.0, audio_len)
            crop_dur = rng.uniform(min_crop, audio_len)
            crop_start = rng.uniform(0, max(0, audio_len - crop_dur))
            merge_clips.append((idx, crop_start, crop_dur))

        merged_total = sum(c[2] for c in merge_clips)
        if LONG_MIN_SEC <= merged_total <= LONG_MAX_SEC:
            break
        # If too short, try extending; if too long, try trimming last
        if merged_total < LONG_MIN_SEC:
            continue
        if merged_total > LONG_MAX_SEC:
            excess = merged_total - LONG_MAX_SEC
            last_idx, last_cs, last_cd = merge_clips[-1]
            if last_cd - excess >= MIN_CROP_SEC:
                merge_clips[-1] = (last_idx, last_cs, last_cd - excess)
                merged_total = LONG_MAX_SEC
                break
    else:
        # Fallback: just use what we have
        pass

    # Pick filler clips to fill remaining time
    target_total = rng.uniform(MIN_TOTAL_SEC, MAX_TOTAL_SEC)
    remaining = target_total - sum(c[2] for c in merge_clips)
    n_fillers = rng.randint(max(1, MIN_CLIPS - n_merge), max(2, MAX_CLIPS - n_merge) + 1)

    filler_clips = []
    for _ in range(n_fillers):
        f_idx = rng.randint(0, len(source_pool))
        src = source_pool[f_idx]
        audio_len = src["audio_len"]
        crop_dur = rng.uniform(MIN_CROP_SEC, min(audio_len, max(MIN_CROP_SEC, remaining / max(1, n_fillers - len(filler_clips)))))
        crop_start = rng.uniform(0, max(0, audio_len - crop_dur))
        filler_clips.append((f_idx, crop_start, crop_dur))
        remaining -= crop_dur
        if remaining <= 0:
            break

    # Decide where to place the long segment (random position among fillers)
    n_before = rng.randint(0, len(filler_clips) + 1)
    before_fillers = filler_clips[:n_before]
    after_fillers = filler_clips[n_before:]

    # Build final sequence: before_fillers + merge_clips + after_fillers
    all_clips = []  # (src_idx, crop_start, crop_dur, is_merge)
    for c in before_fillers:
        all_clips.append((*c, False))
    for c in merge_clips:
        all_clips.append((*c, True))
    for c in after_fillers:
        all_clips.append((*c, False))

    # Build audio and segments
    audio_parts = []
    segments = []
    current_pos = 0.0
    prev_was_merge = False

    for i, (src_idx, crop_start, crop_dur, is_merge) in enumerate(all_clips):
        src = source_pool[src_idx]

        # Determine gap
        if i == 0:
            gap = 0.0
        elif is_merge and prev_was_merge:
            # No gap between merged clips
            gap = 0.0
        else:
            if rng.random() < SILENCE_PROB:
                gap = rng.uniform(SILENCE_MIN_SEC, SILENCE_MAX_SEC)
            else:
                gap = 0.0

        # Add silence gap
        if gap > 0:
            silence_samples = int(round(gap * TARGET_SR))
            audio_parts.append(make_gap_noise(silence_samples, rng))
            current_pos += gap

        # Load and crop audio
        audio_data = load_and_preprocess(src["audio_path"])
        start_sample = int(round(crop_start * TARGET_SR))
        end_sample = start_sample + int(round(crop_dur * TARGET_SR))
        end_sample = min(end_sample, len(audio_data))
        cropped = audio_data[start_sample:end_sample]

        actual_dur = len(cropped) / TARGET_SR
        concat_start = current_pos
        concat_end = current_pos + actual_dur

        audio_parts.append(apply_fade(cropped))
        segments.append(
            SegmentInfo(
                caption=src["caption"],
                crop_start=round(crop_start, 3),
                crop_duration=round(actual_dur, 3),
                concat_start=round(concat_start, 3),
                concat_end=round(concat_end, 3),
                gap_before=round(gap, 3),
                long_group_id=0 if is_merge else None,  # group 0 = the merged event
            )
        )
        current_pos = concat_end
        prev_was_merge = is_merge

    audio = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)

    # Hard truncate
    max_samples = int(round(MAX_TOTAL_SEC * TARGET_SR))
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        if segments and segments[-1].concat_end > MAX_TOTAL_SEC:
            s = segments[-1]
            segments[-1] = s._replace(concat_end=round(MAX_TOTAL_SEC, 3))

    return audio, segments


def generate_multi_sample(
    source_pool: list,
    rng: np.random.RandomState,
) -> tuple:
    """
    Generate a sample where one target event repeats 2-3 times with different
    sounds in between. Target clips are never adjacent.

    Structure: [filler] [TARGET] [filler] [TARGET] [filler] [TARGET] [filler]

    Returns:
        audio: np.ndarray, segments: list of SegmentInfo
    """
    n_repeats = rng.randint(MULTI_REPEAT_MIN, MULTI_REPEAT_MAX + 1)  # 2 or 3

    # Pick target clip
    target_idx = rng.randint(0, len(source_pool))
    target_src = source_pool[target_idx]
    target_caption = target_src["caption"]

    # Build filler pool: must have different caption than target
    filler_pool_indices = [
        i for i, s in enumerate(source_pool) if s["caption"] != target_caption
    ]
    if len(filler_pool_indices) < n_repeats + 1:
        # Fallback: allow any clip that's not the exact same index
        filler_pool_indices = [i for i in range(len(source_pool)) if i != target_idx]

    # Build interleaved sequence: filler, target, filler, target, ..., filler
    sequence = []  # (src_idx, is_target)
    for i in range(n_repeats):
        # filler before this target
        f_idx = filler_pool_indices[rng.randint(0, len(filler_pool_indices))]
        sequence.append((f_idx, False))
        # target (use same source but different crop each time)
        sequence.append((target_idx, True))
    # filler after last target
    f_idx = filler_pool_indices[rng.randint(0, len(filler_pool_indices))]
    sequence.append((f_idx, False))

    # Crop each clip
    clips = []
    for src_idx, _is_target in sequence:
        src = source_pool[src_idx]
        audio_len = src["audio_len"]
        if audio_len < MIN_CROP_SEC:
            crop_dur = audio_len
            crop_start = 0.0
        else:
            crop_dur = rng.uniform(MIN_CROP_SEC, audio_len)
            crop_start = rng.uniform(0, audio_len - crop_dur)
        clips.append((src_idx, crop_start, crop_dur))

    # Adjust total duration to 30-60s
    total = sum(c[2] for c in clips)
    # Add gaps (always gap between clips to separate them)
    gaps = [0.0]  # no gap before first
    for i in range(1, len(clips)):
        gaps.append(rng.uniform(SILENCE_MIN_SEC, SILENCE_MAX_SEC))
    total += sum(gaps)

    # If too long, trim filler clips
    if total > MAX_TOTAL_SEC:
        excess = total - MAX_TOTAL_SEC
        for i in range(len(clips) - 1, -1, -1):
            if not sequence[i][1]:  # is filler
                src_idx, cs, cd = clips[i]
                trim = min(excess, cd - MIN_CROP_SEC)
                if trim > 0:
                    clips[i] = (src_idx, cs, cd - trim)
                    excess -= trim
                if excess <= 0:
                    break

    # If too short, extend filler clips
    total = sum(c[2] for c in clips) + sum(gaps)
    if total < MIN_TOTAL_SEC:
        deficit = MIN_TOTAL_SEC - total
        for i in range(len(clips)):
            if not sequence[i][1]:  # is filler
                src_idx, cs, cd = clips[i]
                src = source_pool[src_idx]
                available = src["audio_len"] - (cs + cd)
                extend = min(available, deficit)
                if extend > 0:
                    clips[i] = (src_idx, cs, cd + extend)
                    deficit -= extend
                if deficit <= 0:
                    break

    # Build audio and segments
    audio_parts = []
    segments = []
    current_pos = 0.0

    for i, (src_idx, crop_start, crop_dur) in enumerate(clips):
        src = source_pool[src_idx]
        gap = gaps[i] if i < len(gaps) else 0.0

        if gap > 0:
            silence_samples = int(round(gap * TARGET_SR))
            audio_parts.append(make_gap_noise(silence_samples, rng))
            current_pos += gap

        audio_data = load_and_preprocess(src["audio_path"])
        start_sample = int(round(crop_start * TARGET_SR))
        end_sample = start_sample + int(round(crop_dur * TARGET_SR))
        end_sample = min(end_sample, len(audio_data))
        cropped = audio_data[start_sample:end_sample]

        actual_dur = len(cropped) / TARGET_SR
        concat_start = current_pos
        concat_end = current_pos + actual_dur

        audio_parts.append(apply_fade(cropped))
        segments.append(
            SegmentInfo(
                caption=src["caption"],
                crop_start=round(crop_start, 3),
                crop_duration=round(actual_dur, 3),
                concat_start=round(concat_start, 3),
                concat_end=round(concat_end, 3),
                gap_before=round(gap, 3),
            )
        )
        current_pos = concat_end

    audio = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)

    # Hard truncate
    max_samples = int(round(MAX_TOTAL_SEC * TARGET_SR))
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        if segments and segments[-1].concat_end > MAX_TOTAL_SEC:
            s = segments[-1]
            segments[-1] = s._replace(concat_end=round(MAX_TOTAL_SEC, 3))

    return audio, segments


# ──────────────────────────── Worker ────────────────────────────

def worker_fn(args):
    """Worker for multiprocessing: generate one sample, save WAV, return metadata."""
    idx, split_name, source_pool, sample_seed, output_dir, prefix, mode = args
    rng = np.random.RandomState(sample_seed)

    sample_id = f"{prefix}_{idx + 1:05d}"
    wav_path = os.path.join(output_dir, "audio", split_name, f"{sample_id}.wav")

    try:
        # Pre-decide target_duration here to check mode feasibility
        target_duration = rng.uniform(MIN_TOTAL_SEC, MAX_TOTAL_SEC)

        # Check mode feasibility based on target_duration
        actual_mode = mode
        if mode == "long" and target_duration < LONG_FEASIBLE_TOTAL:
            actual_mode = "single"  # fallback
        elif mode == "multi":
            # multi needs at least MULTI_FEASIBLE_CLIPS clips, which need ~MULTI_FEASIBLE_CLIPS*MIN_CROP seconds
            min_dur_for_multi = MULTI_FEASIBLE_CLIPS * MIN_CROP_SEC + 1.0
            if target_duration < min_dur_for_multi:
                actual_mode = "single"  # fallback

        if actual_mode == "long":
            audio, segments = generate_long_sample(source_pool, rng)
        elif actual_mode == "multi":
            audio, segments = generate_multi_sample(source_pool, rng)
        else:
            audio, segments = generate_single_sample(source_pool, rng, target_duration=target_duration)
        total_dur = round(len(audio) / TARGET_SR, 3)
        # Skip samples that are too short (require 2+ segments to avoid single moment)
        if total_dur < MIN_TOTAL_SEC or len(segments) < 2:
            return {
                "sample_id": sample_id,
                "mode": actual_mode,
                "error": f"too short: {total_dur}s, {len(segments)} segments",
                "success": False,
            }
        mode = actual_mode  # update for return
        save_wav(wav_path, audio)
        return {
            "sample_id": sample_id,
            "mode": mode,
            "segments": [s._asdict() for s in segments],
            "total_duration": total_dur,
            "success": True,
        }
    except Exception as e:
        return {
            "sample_id": sample_id,
            "mode": mode,
            "error": str(e),
            "success": False,
        }


# ──────────────────────────── Annotation Generation ─────────────

def _resolve_caption_key(caption: str, used_keys: dict) -> str:
    """Handle duplicate caption keys by adding suffix."""
    if caption not in used_keys:
        used_keys[caption] = 1
        return caption
    else:
        used_keys[caption] += 1
        return f"{caption} ({used_keys[caption]})"


def make_grounding_annotation(sample_id: str, segments: list, rng: np.random.RandomState) -> dict:
    templates = [
        "The sound of {event} occurs from {start} to {end} seconds.",
        "{event} is heard from {start} to {end} seconds.",
        "From {start} to {end} seconds, {event} can be heard.",
        "{event} plays between {start} and {end} seconds.",
    ]
    event_dict = {}
    caption_parts = []
    used_keys = {}

    for seg in segments:
        key = _resolve_caption_key(seg["caption"], used_keys)
        s = round(seg["concat_start"], 1)
        e = round(seg["concat_end"], 1)
        event_dict[key] = [[s, e]]
        tmpl = templates[rng.randint(0, len(templates))]
        caption_parts.append(
            tmpl.format(event=key, start=s, end=e)
        )

    return {
        "event": event_dict,
        "caption": " ".join(caption_parts),
    }


def make_ordering_annotation(sample_id: str, segments: list, rng: np.random.RandomState) -> dict:
    event_dict = {}
    used_keys = {}

    # segments are already in temporal order
    keys = []
    for seg in segments:
        key = _resolve_caption_key(seg["caption"], used_keys)
        s = round(seg["concat_start"], 1)
        e = round(seg["concat_end"], 1)
        event_dict[key] = [[s, e]]
        keys.append(key)

    # build caption
    if len(keys) == 1:
        caption = f"Only {keys[0]} was heard."
    elif len(keys) == 2:
        caption = f"{keys[0]} was heard first, followed by {keys[1]}."
    else:
        ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
        parts = []
        for i, k in enumerate(keys):
            if i == 0:
                parts.append(f"{k} was heard first")
            elif i == len(keys) - 1:
                parts.append(f"and finally {k}")
            else:
                if i < len(ordinals):
                    parts.append(f"then {k}")
                else:
                    parts.append(f"then {k}")

        caption = ", ".join(parts) + "."

    return {
        "event": event_dict,
        "caption": caption,
    }


def make_frequency_annotation(sample_id: str, segments: list, rng: np.random.RandomState) -> dict:
    count_words = {1: "once", 2: "twice", 3: "three times", 4: "four times",
                   5: "five times", 6: "six times", 7: "seven times", 8: "eight times"}

    # group by original caption (no suffix) to count occurrences
    grouped = OrderedDict()
    for seg in segments:
        cap = seg["caption"]
        start_time = round(seg["concat_start"], 1)
        if cap not in grouped:
            grouped[cap] = []
        grouped[cap].append(start_time)

    event_dict = {}
    caption_parts = []
    for cap, times in grouped.items():
        event_dict[cap] = times
        n = len(times)
        word = count_words.get(n, f"{n} times")
        caption_parts.append(f"{cap} occurred {word}.")

    return {
        "event": event_dict,
        "caption": " ".join(caption_parts),
    }


def make_duration_annotation(sample_id: str, segments: list, rng: np.random.RandomState) -> dict:
    templates = [
        "{event} lasted {dur} seconds.",
        "{event} continued for {dur} seconds.",
        "The sound of {event} persisted for {dur} seconds.",
        "{event} was heard for {dur} seconds.",
    ]
    event_dict = {}
    caption_parts = []
    used_keys = {}

    for seg in segments:
        key = _resolve_caption_key(seg["caption"], used_keys)
        dur = round(seg["concat_end"] - seg["concat_start"], 1)
        event_dict[key] = [dur]
        tmpl = templates[rng.randint(0, len(templates))]
        caption_parts.append(tmpl.format(event=key, dur=dur))

    return {
        "event": event_dict,
        "caption": " ".join(caption_parts),
    }


# ──────────────────────────── Main Pipeline ─────────────────────

def load_source_annotations(source_dir: str, split: str, annotation_file: str = None) -> list:
    """Load AudioCaps JSON annotation for a split."""
    if annotation_file:
        ann_path = annotation_file
    else:
        split_map = {
            "train": ("train.json", "train"),
            "val": ("val.json", "validation"),
            "test": ("test.json", "test"),
        }
        ann_file, audio_subdir = split_map[split]
        ann_path = os.path.join(source_dir, "annotation", ann_file)

    with open(ann_path, "r") as f:
        annotations = json.load(f)

    # filter out clips shorter than MIN_CROP_SEC
    valid = []
    skipped = 0
    for item in annotations:
        if item["audio_len"] >= MIN_CROP_SEC:
            # verify path exists
            if os.path.exists(item["audio_path"]):
                valid.append(item)
            else:
                skipped += 1
        else:
            skipped += 1

    print(f"  [{split}] Loaded {len(valid)} clips (skipped {skipped}) from {ann_path}")
    return valid


def parse_mode_ratio(mode_ratio_str: str) -> dict:
    """Parse mode ratio string like 'single:0.7,long:0.15,multi:0.15'."""
    if not mode_ratio_str:
        return {"single": 1.0}
    ratios = {}
    for part in mode_ratio_str.split(","):
        mode, ratio = part.strip().split(":")
        ratios[mode.strip()] = float(ratio.strip())
    total = sum(ratios.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Mode ratios must sum to 1.0, got {total}")
    return ratios


def generate_split(
    split: str,
    n_samples: int,
    source_pool: list,
    output_dir: str,
    seed: int,
    workers: int,
    prefix: str = "concat",
    mode_ratio: dict = None,
):
    """Generate all samples for one split with mixed modes."""
    if mode_ratio is None:
        mode_ratio = {"single": 1.0}

    # create audio output dir
    audio_dir = os.path.join(output_dir, "audio", split)
    os.makedirs(audio_dir, exist_ok=True)

    split_offset = {"train": 0, "val": 1, "test": 2}[split]

    # Assign modes to samples based on ratio
    mode_assignments = []
    for mode, ratio in mode_ratio.items():
        n = int(round(n_samples * ratio))
        mode_assignments.extend([mode] * n)
    # Pad/trim to exact n_samples
    while len(mode_assignments) < n_samples:
        mode_assignments.append("single")
    mode_assignments = mode_assignments[:n_samples]
    # Shuffle so modes are interleaved
    mode_rng = np.random.RandomState(seed + split_offset)
    mode_rng.shuffle(mode_assignments)

    # Print mode distribution
    from collections import Counter
    mode_counts = Counter(mode_assignments)
    print(f"  Mode distribution: {dict(mode_counts)}")

    # prepare worker args
    worker_args = []
    for idx in range(n_samples):
        sample_seed = seed * 100000 + split_offset * 50000 + idx
        worker_args.append((idx, split, source_pool, sample_seed, output_dir, prefix, mode_assignments[idx]))

    # run with multiprocessing
    print(f"\n  Generating {n_samples} samples for [{split}] with {workers} workers...")
    results = []

    try:
        from tqdm import tqdm
        with mp.Pool(workers) as pool:
            for result in tqdm(
                pool.imap(worker_fn, worker_args),
                total=n_samples,
                desc=f"  [{split}]",
            ):
                results.append(result)
    except ImportError:
        # fallback without tqdm
        with mp.Pool(workers) as pool:
            for i, result in enumerate(pool.imap(worker_fn, worker_args)):
                results.append(result)
                if (i + 1) % 100 == 0:
                    print(f"    [{split}] {i + 1}/{n_samples}")

    # check results
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    if failures:
        print(f"  [{split}] WARNING: {len(failures)} samples failed!")
        for f in failures[:5]:
            print(f"    {f['sample_id']}: {f['error']}")

    # Save segment metadata (for future annotation regeneration without re-creating audio)
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = os.path.join(metadata_dir, f"{split}_metadata.json")
    metadata = {r["sample_id"]: {"segments": r["segments"], "total_duration": r["total_duration"], "mode": r.get("mode", "single")} for r in successes}
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"    Saved {metadata_path} ({len(metadata)} entries)")

    # Generate grounding annotations only
    print(f"  [{split}] Generating grounding annotations for {len(successes)} samples...")
    grounding_ann = {}

    for r in successes:
        sid = r["sample_id"]
        segs = r["segments"]
        rng = np.random.RandomState(hash(sid) % (2**31))
        grounding_ann[sid] = make_grounding_annotation(sid, segs, rng)

    task_dir = os.path.join(output_dir, f"{split}_grounding")
    os.makedirs(task_dir, exist_ok=True)
    ann_path = os.path.join(task_dir, "grounding_captions.json")
    with open(ann_path, "w") as f:
        json.dump(grounding_ann, f, indent=4)
    print(f"    Saved {ann_path} ({len(grounding_ann)} entries)")

    # Print stats
    durations = [r["total_duration"] for r in successes]
    if durations:
        print(f"  [{split}] Duration stats: "
              f"min={min(durations):.1f}s, max={max(durations):.1f}s, "
              f"mean={np.mean(durations):.1f}s, std={np.std(durations):.1f}s")
        clip_counts = [len(r["segments"]) for r in successes]
        print(f"  [{split}] Clip count stats: "
              f"min={min(clip_counts)}, max={max(clip_counts)}, "
              f"mean={np.mean(clip_counts):.1f}")
        mode_counts = Counter(r.get("mode", "single") for r in successes)
        print(f"  [{split}] Mode stats: {dict(mode_counts)}")

    return successes


def regenerate_annotations_from_metadata(output_dir: str, splits: list):
    """Regenerate annotation JSONs from saved metadata (no audio generation needed)."""
    for split in splits:
        metadata_path = os.path.join(output_dir, "metadata", f"{split}_metadata.json")
        if not os.path.exists(metadata_path):
            print(f"  [{split}] Metadata not found at {metadata_path}, skipping.")
            continue

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        print(f"  [{split}] Regenerating grounding annotations from metadata ({len(metadata)} samples)...")
        grounding_ann = {}

        for sid, meta in metadata.items():
            segs = meta["segments"]
            rng = np.random.RandomState(hash(sid) % (2**31))
            grounding_ann[sid] = make_grounding_annotation(sid, segs, rng)

        task_dir = os.path.join(output_dir, f"{split}_grounding")
        os.makedirs(task_dir, exist_ok=True)
        ann_path = os.path.join(task_dir, "grounding_captions.json")
        with open(ann_path, "w") as f:
            json.dump(grounding_ann, f, indent=4)
        print(f"    Saved {ann_path} ({len(grounding_ann)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Generate AudioCapsConcat dataset")
    parser.add_argument("--source_dir", type=str, default=None,
                        help="Path to AudioCaps dataset root (not needed for --regen_only)")
    parser.add_argument("--annotation_file", type=str, default=None,
                        help="Override annotation JSON (e.g. crop_safe.json). Used for all splits.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output AudioCapsConcat dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--regen_only", action="store_true",
                        help="Only regenerate annotations from saved metadata (no audio generation)")
    parser.add_argument("--prefix", type=str, default="concat",
                        help="Filename prefix for samples (e.g., 'grounding', 'duration')")
    parser.add_argument("--mode_ratio", type=str, default=None,
                        help="Mode ratio string, e.g. 'single:0.7,long:0.15,multi:0.15'. "
                             "Default: single:1.0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Annotation-only mode: regenerate from metadata
    if args.regen_only:
        print(f"Regenerating annotations from metadata in {args.output_dir}")
        regenerate_annotations_from_metadata(args.output_dir, args.splits)
        print("Done!")
        return

    if not args.source_dir:
        parser.error("--source_dir is required when not using --regen_only")

    mode_ratio = parse_mode_ratio(args.mode_ratio)

    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}, Workers: {args.workers}")
    print(f"Mode ratio: {mode_ratio}")

    split_samples = {
        "train": args.train_samples,
        "val": args.val_samples,
        "test": args.test_samples,
    }

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}")

        source_pool = load_source_annotations(args.source_dir, split, args.annotation_file)
        n_samples = split_samples[split]

        generate_split(
            split=split,
            n_samples=n_samples,
            source_pool=source_pool,
            output_dir=args.output_dir,
            seed=args.seed,
            workers=args.workers,
            prefix=args.prefix,
            mode_ratio=mode_ratio,
        )

    print(f"\nDone! Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()
