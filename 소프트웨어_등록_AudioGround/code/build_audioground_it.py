#!/usr/bin/env python3
"""
Build AudioGround-IT annotation JSON from AudioCapsConcat_paper data.

Paper structure (Table 2): each task uses SEPARATE audio files.
  Files 0..16199   → grounding  (16.2K, 1 sample/file, primary event)
  Files 16200..31399 → duration  (15.2K, 1 sample/file, one segment)
  Files 31400..40699 → ordering  (9.3K,  1 sample/file, one pair)
  V2 data          → frequency  (9.1K)

Total audio: ~40.5K files × ~70s avg ≈ 787h ≈ paper 835h
"""
import json, os, random, argparse
from collections import defaultdict, Counter
from itertools import combinations

random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
META      = "/home/irteam/avqa2/SALMONN/data/AudioCapsConcat_paper/metadata/train_metadata.json"
AUDIO_DIR = "/home/irteam/avqa2/SALMONN/data/AudioCapsConcat_paper/audio/train"
V2_ANN    = "/home/irteam/Qwen2Audio-GRPO/_DATASETS/AudioCapsConcat_v2/temporal_reasoning_v2_50k.json"
V2_AUDIO  = "/home/irteam/Qwen2Audio-GRPO/_DATASETS/AudioCapsConcat_v2/audio/train"
OUT_PATH  = "/home/irteam/avqa2/SALMONN/data/audioground_it_train.json"

# ── File index splits (paper: separate audio per task) ─────────────────────────
GRD_END = 16200   # files [0, 16200)  → grounding
DUR_END = 31400   # files [16200, 31400) → duration
ORD_END = 40700   # files [31400, 40700) → ordering

# ── Paper targets (Table 2) ────────────────────────────────────────────────────
TARGET = {
    "grounding": 16200,
    "duration":  15200,
    "ordering":   9300,
    "frequency":  9100,
}

# ── Question templates ─────────────────────────────────────────────────────────
GRD_Q = [
    "When does '{cap}' occur?",
    "At what time does '{cap}' happen?",
    "Identify the time segment where '{cap}' is present.",
    "Find the time interval of '{cap}'.",
    "Locate '{cap}' in the audio.",
]
DUR_Q = [
    "How long does '{cap}' last?",
    "What is the duration of '{cap}'?",
    "How many seconds does '{cap}' last?",
]
ORD_Q = [
    "Which sound occurs first, '{cap1}' or '{cap2}'?",
    "Which sound event happened earlier, '{cap1}' or '{cap2}'?",
    "Between '{cap1}' and '{cap2}', which one comes first?",
    "Does '{cap1}' or '{cap2}' occur first in the audio?",
]
FRQ_Q = [
    "How many times does '{cap}' occur?",
    "How often does '{cap}' appear in the audio?",
    "Count the occurrences of '{cap}' in the audio.",
    "How many occurrences of '{cap}' are there?",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def fix_path(p: str) -> str:
    return p.replace("/home1/irteam/", "/home/irteam/")

def pos_bin(start: float, total: float, n: int = 5) -> int:
    if total <= 0:
        return 0
    return min(int((start / total) * n), n - 1)

def balance_by_pos(samples: list, n_target: int, n_bins: int = 5) -> list:
    bins = defaultdict(list)
    for s in samples:
        bins[s["_bin"]].append(s)

    per_bin = n_target // n_bins
    result, used_ids = [], set()

    for b in range(n_bins):
        pool = bins.get(b, [])
        take = min(per_bin, len(pool))
        chosen = random.sample(pool, take)
        result.extend(chosen)
        used_ids.update(id(x) for x in chosen)

    shortfall = n_target - len(result)
    if shortfall > 0:
        leftover = [s for s in samples if id(s) not in used_ids]
        if leftover:
            extra = random.sample(leftover, min(shortfall, len(leftover)))
            result.extend(extra)

    random.shuffle(result)
    return result[:n_target]


def _primary_event(segs: list):
    """
    Pick the primary grounding event from a file's segments.
    Prefer the longest contiguous long_group event; fall back to longest segment.
    Returns (start, end, caption).
    """
    # Build merged groups
    groups = []
    i = 0
    while i < len(segs):
        gid = segs[i].get("long_group_id")
        if gid is not None:
            group = [segs[i]]
            j = i + 1
            while j < len(segs) and segs[j].get("long_group_id") == gid:
                group.append(segs[j])
                j += 1
            groups.append((group[0]["concat_start"], group[-1]["concat_end"],
                           group[0]["caption"], True))
            i = j
        else:
            s = segs[i]
            groups.append((s["concat_start"], s["concat_end"], s["caption"], False))
            i += 1

    if not groups:
        return None

    # Prefer long_group events (merged), else longest individual segment
    long_groups = [(s, e, c) for s, e, c, is_long in groups if is_long]
    if long_groups:
        best = max(long_groups, key=lambda x: x[1] - x[0])
        return best
    else:
        all_events = [(s, e, c) for s, e, c, _ in groups]
        return max(all_events, key=lambda x: x[1] - x[0])


# ── Generators (1 sample per file) ─────────────────────────────────────────────

def gen_grounding(meta_slice: dict) -> list:
    samples = []
    for fname, entry in meta_slice.items():
        path  = os.path.join(AUDIO_DIR, f"{fname}.wav")
        total = entry["total_duration"]
        segs  = entry["segments"]
        event = _primary_event(segs)
        if event is None:
            continue
        start, end, cap = event
        q   = random.choice(GRD_Q).format(cap=cap)
        txt = f"The given query occurs at {start:.1f} to {end:.1f} seconds."
        samples.append(dict(
            task="grounding", path=path, total_duration=total,
            Q=q, text=txt, _bin=pos_bin(start, total),
        ))
    return samples


def gen_duration(meta_slice: dict) -> list:
    samples = []
    for fname, entry in meta_slice.items():
        path  = os.path.join(AUDIO_DIR, f"{fname}.wav")
        total = entry["total_duration"]
        segs  = entry["segments"]
        if not segs:
            continue
        seg = random.choice(segs)
        start, end = seg["concat_start"], seg["concat_end"]
        dur = end - start
        cap = seg["caption"]
        q   = random.choice(DUR_Q).format(cap=cap)
        txt = f"It lasts {dur:.1f} seconds, from {start:.1f} to {end:.1f} seconds."
        samples.append(dict(
            task="duration", path=path, total_duration=total,
            Q=q, text=txt, _bin=pos_bin(start, total),
        ))
    return samples


def gen_ordering(meta_slice: dict) -> list:
    samples = []
    for fname, entry in meta_slice.items():
        path  = os.path.join(AUDIO_DIR, f"{fname}.wav")
        total = entry["total_duration"]
        segs  = entry["segments"]
        if len(segs) < 2:
            continue
        # pick one random pair of distinct-caption segments
        distinct = [s for s in segs if s.get("long_group_id") is None]
        if len(distinct) < 2:
            distinct = segs
        if len(distinct) < 2:
            continue
        s1, s2 = random.sample(distinct, 2)
        first, second = (
            (s1, s2) if s1["concat_start"] <= s2["concat_start"] else (s2, s1)
        )
        cap_f, cap_s = first["caption"], second["caption"]
        if random.random() < 0.5:
            q = random.choice(ORD_Q).format(cap1=cap_f, cap2=cap_s)
        else:
            q = random.choice(ORD_Q).format(cap1=cap_s, cap2=cap_f)
        txt = (
            f"'{cap_f}' occurs first, "
            f"from {first['concat_start']:.1f} to {first['concat_end']:.1f} seconds."
        )
        samples.append(dict(
            task="ordering", path=path, total_duration=total,
            Q=q, text=txt, _bin=pos_bin(first["concat_start"], total),
        ))
    return samples


def gen_frequency(v2_ann: list) -> list:
    samples = []
    for ann in v2_ann:
        if ann.get("reasoning_type") not in ("counting", "repetition"):
            continue
        if ann.get("is_negative", False):
            continue

        seg_info = ann.get("segments_info", [])
        total    = ann["total_duration"]
        path     = fix_path(ann["path"])

        cap_segs = defaultdict(list)
        for s in seg_info:
            cap_segs[s["caption"]].append((s["start"], s["end"]))

        for cap, occurrences in cap_segs.items():
            if len(occurrences) < 2:
                continue
            occurrences = sorted(occurrences)
            N = len(occurrences)
            parts = [f"{s:.1f} to {e:.1f} seconds" for s, e in occurrences]
            if N == 2:
                interval_str = f"{parts[0]}, and {parts[1]}"
            else:
                interval_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"
            q   = random.choice(FRQ_Q).format(cap=cap)
            txt = f"It occurs {N} times, at {interval_str}."
            samples.append(dict(
                task="frequency", path=path, total_duration=total,
                Q=q, text=txt, _bin=pos_bin(occurrences[0][0], total),
            ))
    return samples


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading metadata...")
    all_meta = json.load(open(META))
    v2_ann   = json.load(open(V2_ANN))["annotation"]

    # Sort by filename (concat_00001 … concat_40500) for deterministic split
    all_keys = sorted(all_meta.keys())
    print(f"  Total audio files: {len(all_keys)}")

    grd_keys = all_keys[:GRD_END]
    dur_keys = all_keys[GRD_END:DUR_END]
    ord_keys = all_keys[DUR_END:ORD_END]

    grd_meta = {k: all_meta[k] for k in grd_keys if k in all_meta}
    dur_meta = {k: all_meta[k] for k in dur_keys if k in all_meta}
    ord_meta = {k: all_meta[k] for k in ord_keys if k in all_meta}

    print(f"  Grounding pool : {len(grd_meta):,} files")
    print(f"  Duration  pool : {len(dur_meta):,} files")
    print(f"  Ordering  pool : {len(ord_meta):,} files")

    print("Generating candidates...")
    grd_pool = gen_grounding(grd_meta)
    dur_pool = gen_duration(dur_meta)
    ord_pool = gen_ordering(ord_meta)
    frq_pool = gen_frequency(v2_ann)

    print(f"  grounding pool : {len(grd_pool):>7,}")
    print(f"  duration  pool : {len(dur_pool):>7,}")
    print(f"  ordering  pool : {len(ord_pool):>7,}")
    print(f"  frequency pool : {len(frq_pool):>7,}")

    print("Applying temporal-position bias control...")
    grd  = balance_by_pos(grd_pool,  TARGET["grounding"])
    dur  = balance_by_pos(dur_pool,  TARGET["duration"])
    ord_ = balance_by_pos(ord_pool,  TARGET["ordering"])
    frq  = balance_by_pos(frq_pool,  min(TARGET["frequency"], len(frq_pool)))

    print(f"  grounding : {len(grd):,}  (target {TARGET['grounding']:,})")
    print(f"  duration  : {len(dur):,}  (target {TARGET['duration']:,})")
    print(f"  ordering  : {len(ord_):,}  (target {TARGET['ordering']:,})")
    print(f"  frequency : {len(frq):,}  (target {TARGET['frequency']:,})")

    annotation = []
    for task_samples in (grd, dur, ord_, frq):
        for s in task_samples:
            s.pop("_bin", None)
            annotation.append(s)

    random.shuffle(annotation)

    out = {"annotation": annotation}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(annotation):,} samples → {args.out}")

    task_counts = Counter(a["task"] for a in annotation)
    dur_vals    = [a["total_duration"] for a in annotation]
    print("\nTask breakdown:")
    for t, n in sorted(task_counts.items()):
        print(f"  {t:12s}: {n:,}")
    print(f"\nAudio duration: min={min(dur_vals):.1f}s  "
          f"max={max(dur_vals):.1f}s  "
          f"mean={sum(dur_vals)/len(dur_vals):.1f}s")
    print(f"Total unique audio: ~{len(set(a['path'] for a in annotation)):,} files")


if __name__ == "__main__":
    main()
