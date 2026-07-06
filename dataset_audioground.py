"""AudioGround dataset: handles audio longer than 30 s by chunking."""

import json

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor


class AudioGroundDataset(Dataset):
    """Dataset for AudioGround-IT with multi-chunk spectrogram support.

    Audio longer than CHUNK_SIZE seconds is split into ceil(dur/CHUNK_SIZE)
    non-overlapping 30 s chunks. Each chunk is processed by Whisper's feature
    extractor independently and stacked into [N_chunks, 128, 3000].

    All samples are zero-padded to max_duration so every item has the same
    N_chunks (= max_duration // CHUNK_SIZE = 2 for 60 s max), which allows
    simple torch.stack collation without custom padding logic.
    """

    CHUNK_SEC  = 30          # Whisper's fixed context window
    MAX_SEC    = 120         # paper: up to 120s audio
    SR         = 16_000      # target sample rate

    def __init__(self, ann_path: str, whisper_path: str):
        super().__init__()
        self.annotation  = json.load(open(ann_path))["annotation"]
        self.wav_proc    = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.n_chunks    = self.MAX_SEC // self.CHUNK_SEC   # 4
        self.chunk_samp  = self.CHUNK_SEC * self.SR         # 480 000 samples
        self.max_samp    = self.MAX_SEC   * self.SR         # 1 920 000 samples

    def __len__(self):
        return len(self.annotation)

    # ── Collater ──────────────────────────────────────────────────────────────

    def collater(self, samples):
        # spectrogram: every item is [N_chunks, 128, 3000] → stack → [B, N, 128, 3000]
        spectrogram = torch.stack([s["spectrogram"] for s in samples], dim=0)

        raw_wav        = [torch.from_numpy(s["raw_wav"]) for s in samples]
        raw_wav_length = torch.tensor([s["raw_wav"].shape[0] for s in samples])
        raw_wav        = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        padding_mask   = (
            torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)
        )

        return {
            "spectrogram":  spectrogram,
            "raw_wav":      raw_wav,
            "padding_mask": padding_mask,
            "text":  [s["text"]  for s in samples],
            "task":  [s["task"]  for s in samples],
            "Q":     [s["Q"]     for s in samples],
            "id":    [s["id"]    for s in samples],
        }

    # ── Item loader ───────────────────────────────────────────────────────────

    def __getitem__(self, index):
        ann = self.annotation[index]

        # Load + mono
        audio, sr = sf.read(ann["path"])
        if audio.ndim == 2:
            audio = audio[:, 0]

        # Resample if needed (simple decimation/repeat; librosa not required)
        if sr != self.SR:
            ratio = self.SR / sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

        # Ensure at least 1 second of content
        if len(audio) < self.SR:
            audio = np.pad(audio, (0, self.SR - len(audio)))

        # Pad (or truncate) to max_samp with silence
        audio_padded = np.zeros(self.max_samp, dtype=np.float32)
        n_copy = min(len(audio), self.max_samp)
        audio_padded[:n_copy] = audio[:n_copy]

        # Build chunk spectrograms: [N_chunks, 128, 3000]
        specs = []
        for i in range(self.n_chunks):
            chunk = audio_padded[i * self.chunk_samp: (i + 1) * self.chunk_samp]
            feat  = self.wav_proc(
                chunk, sampling_rate=self.SR, return_tensors="pt"
            )["input_features"].squeeze(0)              # [128, 3000]
            specs.append(feat)
        spectrogram = torch.stack(specs, dim=0)         # [N_chunks, 128, 3000]

        return {
            "spectrogram": spectrogram,
            "raw_wav":     audio_padded,
            "text":        ann["text"],
            "task":        ann.get("task", "grounding"),
            "Q":           ann.get("Q", ""),
            "id":          ann["path"],
        }
