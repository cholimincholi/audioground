import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

from .salmonn import SALMONN


class AbsoluteTimeEmbedding(nn.Module):
    """Window-level absolute time embedding.

    Three variants (ate_type):
      'sinusoidal' – fixed sinusoidal encoding; no learnable params beyond beta
      'learned'    – MLP from raw scalar time to d_llm
      'hybrid'     – sinusoidal base + zero-init residual MLP  (paper default)
    """

    def __init__(self, d_llm: int, ate_type: str = 'hybrid'):
        super().__init__()
        assert ate_type in ('sinusoidal', 'learned', 'hybrid'), \
            f"ate_type must be one of 'sinusoidal', 'learned', 'hybrid', got '{ate_type}'"
        self.d_llm = d_llm
        self.ate_type = ate_type

        if ate_type == 'learned':
            self.mlp = nn.Sequential(
                nn.Linear(1, d_llm),
                nn.SiLU(),
                nn.Linear(d_llm, d_llm),
            )
        elif ate_type == 'hybrid':
            self.W1 = nn.Linear(d_llm, d_llm)
            self.act = nn.SiLU()
            self.W2 = nn.Linear(d_llm, d_llm)
            # Zero-init so the hybrid starts identical to pure sinusoidal
            nn.init.zeros_(self.W2.weight)
            nn.init.zeros_(self.W2.bias)

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        """Standard transformer sinusoidal encoding.

        TE_sin(c)_{2k}   = sin(omega_k * c)
        TE_sin(c)_{2k+1} = cos(omega_k * c)
        omega_k = 10000^(-2k / d_llm)

        Args:
            t: [N] float tensor of times in seconds
        Returns:
            [N, d_llm]
        """
        half = self.d_llm // 2
        k = torch.arange(half, device=t.device, dtype=t.dtype)
        omega = 10000.0 ** (-2.0 * k / self.d_llm)          # [half]
        angles = t.unsqueeze(1) * omega.unsqueeze(0)          # [N, half]
        # interleave sin and cos: [sin_0, cos_0, sin_1, cos_1, ...]
        enc = torch.stack([torch.sin(angles), torch.cos(angles)], dim=2)
        return enc.reshape(t.shape[0], self.d_llm)            # [N, d_llm]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [N] float tensor of center times in seconds
        Returns:
            [N, d_llm] time embeddings
        """
        if self.ate_type == 'sinusoidal':
            return self._sinusoidal(t)

        if self.ate_type == 'learned':
            return self.mlp(t.unsqueeze(1))   # [N, 1] → [N, d_llm]

        # hybrid: sinusoidal base + zero-init residual MLP
        base = self._sinusoidal(t)                            # [N, d_llm]
        return base + self.W2(self.act(self.W1(base)))        # W2 starts at zero


class AudioGround(SALMONN):
    """SALMONN extended with three temporal-awareness components.

    All three components are independently toggleable for ablation:

      use_frame_interpolation     (FI)  – interpolate BEATs to Whisper frame length
                                          instead of zero-padding
      use_timestamp_conditioning  (TS)  – inject per-window text timestamps into
                                          Q-Former as additional input tokens
      use_absolute_time_embedding (ATE) – add explicit continuous time embedding
                                          to Q-Former output before LLM

    Ablation table from the paper (Table 4):
      FI=T, TS=T, ATE=F                 → SALMONN + FI + TimeStamp
      FI=T, TS=T, ATE=T, type=sinusoidal → + ATEs
      FI=T, TS=T, ATE=T, type=learned    → + ATEl
      FI=T, TS=T, ATE=T, type=hybrid     → + ATEh  ← paper default (Ours)
    """

    def __init__(
        self,
        # ── SALMONN base args ──────────────────────────────────────────────
        llama_path="",
        whisper_path="",
        freeze_whisper=True,
        beats_path="",
        freeze_beats=True,
        use_speech_Qformer=True,
        num_speech_query_token=1,
        freeze_speech_QFormer=False,
        window_level_Qformer=True,
        second_per_window=0.333333,
        second_stride=0.333333,
        speech_llama_proj_model="",
        freeze_speech_llama_proj=False,
        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        multi_prompt=False,
        prompt_path="",
        prompt_template="",
        max_txt_len=128,
        end_sym="</s>",
        low_resource=False,
        device_8bit=0,
        # ── AudioGround-specific args ──────────────────────────────────────
        use_frame_interpolation=True,
        use_timestamp_conditioning=True,
        use_absolute_time_embedding=True,
        ate_type='hybrid',
        ate_beta_init=0.05,
    ):
        super().__init__(
            llama_path=llama_path,
            whisper_path=whisper_path,
            freeze_whisper=freeze_whisper,
            beats_path=beats_path,
            freeze_beats=freeze_beats,
            use_speech_Qformer=use_speech_Qformer,
            num_speech_query_token=num_speech_query_token,
            freeze_speech_QFormer=freeze_speech_QFormer,
            window_level_Qformer=window_level_Qformer,
            second_per_window=second_per_window,
            second_stride=second_stride,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        assert window_level_Qformer, "AudioGround requires window_level_Qformer=True"

        self.use_frame_interpolation = use_frame_interpolation
        self.use_timestamp_conditioning = use_timestamp_conditioning
        self.use_absolute_time_embedding = use_absolute_time_embedding

        # ── Timestamp conditioning ─────────────────────────────────────────
        # SALMONN.__init__ nulls out word_embeddings, position_embeddings,
        # and the text-side FFN (intermediate, output) of every Q-Former layer
        # because the base model never uses text tokens.
        # Restore everything so timestamp text tokens can pass through.
        if use_timestamp_conditioning:
            from transformers import BertModel
            bert_pretrained = BertModel.from_pretrained("bert-base-uncased")

            # Restore text-side FFN from BERT pretrained weights.
            # SALMONN.__init__ sets these to None (query-only Q-Former).
            # We copy from BERT so weights are stable (not random).
            from .Qformer import BertIntermediate, BertOutput
            qf_config = self.speech_Qformer.bert.config
            for i, layer in enumerate(self.speech_Qformer.bert.encoder.layer):
                if layer.intermediate is None:
                    layer.intermediate = BertIntermediate(qf_config)
                    layer.intermediate.load_state_dict(
                        bert_pretrained.encoder.layer[i].intermediate.state_dict()
                    )
                if layer.output is None:
                    layer.output = BertOutput(qf_config)
                    layer.output.load_state_dict(
                        bert_pretrained.encoder.layer[i].output.state_dict()
                    )

            self.speech_Qformer.bert.embeddings.word_embeddings = (
                bert_pretrained.embeddings.word_embeddings
            )
            self.speech_Qformer.bert.embeddings.position_embeddings = (
                bert_pretrained.embeddings.position_embeddings
            )
            del bert_pretrained
            # Propagate freeze setting to the restored embeddings
            if freeze_speech_QFormer:
                self.speech_Qformer.bert.embeddings.word_embeddings.requires_grad_(False)
                self.speech_Qformer.bert.embeddings.position_embeddings.requires_grad_(False)
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            logging.info("AudioGround: timestamp conditioning enabled (pretrained BERT embeddings)")

        # ── Absolute time embedding ────────────────────────────────────────
        if use_absolute_time_embedding:
            d_llm = self.llama_model.config.hidden_size
            self.ate = AbsoluteTimeEmbedding(d_llm, ate_type=ate_type)
            # Learnable scaling factor β, initialized to ate_beta_init (paper: 0.05)
            self.ate_beta = nn.Parameter(torch.tensor(float(ate_beta_init)))
            logging.info(f"AudioGround: absolute time embedding enabled (type={ate_type}, β_init={ate_beta_init})")

    # ── encode_speech: handles chunked [B, N, mel, time] spectrograms ────────

    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        if spectrogram.dim() == 4:
            # Chunked long audio: [B, N_chunks, mel, time]
            B, N, mel, time = spectrogram.shape
            with self.maybe_autocast():
                # Process all chunks through Whisper in one batched call
                speech_flat = spectrogram.view(B * N, mel, time)
                embeds_flat = self.speech_encoder(
                    speech_flat, return_dict=True
                ).last_hidden_state                             # [B*N, T_w, d_w]
                _, T_w, d_w = embeds_flat.shape
                # Concatenate chunk features along time: [B, N*T_w, d_w]
                speech_embeds = embeds_flat.view(B, N * T_w, d_w)

            # BEATs overflows in fp16 — force fp32 by disabling AMP
            if self.beats_path and raw_wav is not None:
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    audio_embeds, _ = self.beats.extract_features(
                        raw_wav.float(),
                        padding_mask=audio_padding_mask,
                        feature_only=True,
                    )
            else:
                audio_embeds = None

            return self._encode_auditory_feature(speech_embeds, audio_embeds)
        else:
            # Standard 30 s path: delegate to base class
            return super().encode_speech(spectrogram, raw_wav, audio_padding_mask)

    # ── Core override ──────────────────────────────────────────────────────

    def _encode_auditory_feature(self, speech_embeds, audio_embeds=None):
        with self.maybe_autocast():
            speech_embeds = self.ln_speech(speech_embeds)

            if audio_embeds is not None:
                audio_embeds = self.ln_audio(audio_embeds)

                # [1] Frame-level alignment: interpolate BEATs to Whisper length
                if self.use_frame_interpolation:
                    if audio_embeds.size(1) != speech_embeds.size(1):
                        audio_embeds = F.interpolate(
                            audio_embeds.transpose(1, 2).float(),
                            size=speech_embeds.size(1),
                            mode='linear',
                            align_corners=False,
                        ).to(speech_embeds.dtype).transpose(1, 2)
                else:
                    # Original SALMONN: zero-pad the shorter side
                    diff = speech_embeds.size(1) - audio_embeds.size(1)
                    if diff > 0:
                        audio_embeds = F.pad(audio_embeds, (0, 0, 0, diff))
                    elif diff < 0:
                        speech_embeds = F.pad(speech_embeds, (0, 0, 0, -diff))

                speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)

            speech_atts = torch.ones(
                speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device
            )

            # ── Sliding-window unfold ──────────────────────────────────────
            B, T, C = speech_embeds.shape
            kernel = round(1500 * self.second_per_window / 30.0)
            stride = round(1500 * self.second_stride / 30.0)

            # [B, C, 1, T] → unfold → [B, C*kernel, L]
            speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
            speech_embeds_unf = F.unfold(
                speech_embeds_tr,
                kernel_size=(1, kernel),
                dilation=1,
                padding=0,
                stride=(1, stride),
            )
            _, _, L = speech_embeds_unf.shape
            # [B, C*kernel, L] → [B, C, kernel, L] → [B, L, kernel, C] → [B*L, kernel, C]
            speech_embeds_unf = speech_embeds_unf.view(B, C, kernel, L)
            speech_embeds_unf = speech_embeds_unf.permute(0, 3, 2, 1)
            speech_wins = speech_embeds_unf.reshape(B * L, kernel, C)
            win_atts = torch.ones(
                speech_wins.size()[:-1], dtype=torch.long, device=speech_wins.device
            )

            NQ = self.speech_query_tokens.shape[1]
            query_tokens = self.speech_query_tokens.expand(B * L, -1, -1)  # [B*L, NQ, H_q]

            # [2] Timestamp conditioning ───────────────────────────────────
            if self.use_timestamp_conditioning:
                # One text per window; same timestamps for every batch element
                ts_texts = [
                    f"This segment is from {l * self.second_stride:.2f} to "
                    f"{l * self.second_stride + self.second_per_window:.2f} seconds, "
                    for l in range(L)
                ]
                ts_enc = self.bert_tokenizer(
                    ts_texts, return_tensors='pt', padding=True
                ).to(speech_wins.device)
                # Tile L texts → B*L: layout matches [B*L] = b*L+l ordering
                ts_ids = ts_enc.input_ids.repeat(B, 1)         # [B*L, T_ts]
                ts_mask = ts_enc.attention_mask.repeat(B, 1)   # [B*L, T_ts]

                # Attention mask covers query tokens (all ones) + text tokens
                qformer_atts = torch.cat([
                    torch.ones(B * L, NQ, device=speech_wins.device, dtype=torch.long),
                    ts_mask,
                ], dim=1)  # [B*L, NQ + T_ts]

                qformer_out = self.speech_Qformer.bert(
                    input_ids=ts_ids,
                    attention_mask=qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=speech_wins,
                    encoder_attention_mask=win_atts,
                    return_dict=True,
                )
                # last_hidden_state: [B*L, NQ + T_ts, H_q] → take only query positions
                H = qformer_out.last_hidden_state[:, :NQ, :]   # [B*L, NQ, H_q]
            else:
                qformer_out = self.speech_Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=speech_wins,
                    encoder_attention_mask=win_atts,
                    return_dict=True,
                )
                H = qformer_out.last_hidden_state               # [B*L, NQ, H_q]

            # Project to LLM hidden space — still un-reshaped at [B*L, NQ, d_llm]
            H = self.speech_llama_proj(H)                       # [B*L, NQ, d_llm]

            # [3] Absolute time embedding ──────────────────────────────────
            # Inject per-window before reshape so window identity is preserved.
            if self.use_absolute_time_embedding:
                center_times = torch.tensor(
                    [l * self.second_stride + self.second_per_window / 2 for l in range(L)],
                    device=H.device,
                    dtype=torch.float32,
                )                                               # [L]
                ates = self.ate(center_times).to(H.dtype)       # [L, d_llm]

                # H: [B*L, NQ, d_llm] → [B, L, NQ, d_llm]
                d = H.size(2)
                H = H.view(B, L, NQ, d)
                # ates: [L, d_llm] → [1, L, 1, d_llm]  broadcast over B and NQ
                H = H + self.ate_beta * ates.unsqueeze(0).unsqueeze(2)
                H = H.reshape(B * L, NQ, d)

            # Final reshape: [B*L, NQ, d_llm] → [B, L*NQ, d_llm]
            speech_embeds_out = H.view(B, -1, H.size(2)).contiguous()
            speech_atts_out = torch.ones(
                speech_embeds_out.size()[:-1], dtype=torch.long, device=speech_embeds_out.device
            )

        return speech_embeds_out, speech_atts_out

    # ── Config loader ──────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config):
        # SALMONN base args
        llama_path               = config.get("llama_path")
        whisper_path             = config.get("whisper_path")
        freeze_whisper           = config.get("freeze_whisper", True)
        beats_path               = config.get("beats_path", "")
        freeze_beats             = config.get("freeze_beats", True)
        use_speech_Qformer       = config.get("use_speech_Qformer", True)
        num_speech_query_token   = config.get("num_speech_query_token", 1)
        freeze_speech_QFormer    = config.get("freeze_speech_QFormer", False)
        window_level_Qformer     = config.get("window_level_Qformer", True)
        second_per_window        = config.get("second_per_window", 0.333333)
        second_stride            = config.get("second_stride", 0.333333)
        speech_llama_proj_model  = config.get("speech_llama_proj_model", "")
        freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)
        lora                     = config.get("lora", True)
        lora_rank                = config.get("lora_rank", 8)
        lora_alpha               = config.get("lora_alpha", 32)
        lora_dropout             = config.get("lora_dropout", 0.1)
        multi_prompt             = config.get("multi_prompt", False)
        prompt_path              = config.get("prompt_path", "")
        prompt_template          = config.get("prompt_template", "")
        max_txt_len              = config.get("max_txt_len", 128)
        end_sym                  = config.get("end_sym", "</s>")
        low_resource             = config.get("low_resource", False)
        device_8bit              = config.get("device_8bit", 0)

        # AudioGround-specific args
        use_frame_interpolation      = config.get("use_frame_interpolation", True)
        use_timestamp_conditioning   = config.get("use_timestamp_conditioning", True)
        use_absolute_time_embedding  = config.get("use_absolute_time_embedding", True)
        ate_type                     = config.get("ate_type", "hybrid")
        ate_beta_init                = config.get("ate_beta_init", 0.05)

        model = cls(
            llama_path=llama_path,
            whisper_path=whisper_path,
            freeze_whisper=freeze_whisper,
            beats_path=beats_path,
            freeze_beats=freeze_beats,
            use_speech_Qformer=use_speech_Qformer,
            num_speech_query_token=num_speech_query_token,
            freeze_speech_QFormer=freeze_speech_QFormer,
            window_level_Qformer=window_level_Qformer,
            second_per_window=second_per_window,
            second_stride=second_stride,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            use_frame_interpolation=use_frame_interpolation,
            use_timestamp_conditioning=use_timestamp_conditioning,
            use_absolute_time_embedding=use_absolute_time_embedding,
            ate_type=ate_type,
            ate_beta_init=ate_beta_init,
        )

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logging.info(f"AudioGround: loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt['model']

            # Skip LoRA keys — they were merged into llama_path before training.
            # Also skip any remaining shape mismatches for safety.
            model_state = model.state_dict()
            filtered, skipped = {}, []
            for k, v in state.items():
                if 'lora' in k.lower():
                    skipped.append(k)
                elif k in model_state and model_state[k].shape == v.shape:
                    filtered[k] = v
                else:
                    skipped.append(k)

            if skipped:
                logging.info(f"AudioGround ckpt: skipped {len(skipped)} keys "
                             f"(LoRA already merged or shape mismatch)")
            model.load_state_dict(filtered, strict=False)
            logging.info(f"AudioGround ckpt: loaded {len(filtered)} / {len(state)} keys")

        return model
