"""
Merge SALMONN-7B's rank-8 LoRA weights into the Vicuna-7B-v1.5 base model
and save a stand-alone merged LLaMA model.

Usage:
    python merge_salmonn_lora.py --out /path/to/salmonn-7b-merged
"""
import argparse
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, get_peft_model

VICUNA_PATH = "/home1/irteam/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5"
SALMONN_CKPT = "/home/irteam/avqa2/SALMONN/ckpts/salmonn_7b_v0.pth"
DEFAULT_OUT  = "/home1/irteam/avqa2/SALMONN/ckpts/salmonn-7b-llama-merged"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    print("Loading Vicuna-7B-v1.5...")
    model = LlamaForCausalLM.from_pretrained(
        VICUNA_PATH, torch_dtype=torch.float16, device_map="cpu"
    )

    # Apply rank-8 LoRA (matching what SALMONN-7B was trained with)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Extract only the LLaMA LoRA keys from the SALMONN checkpoint
    print("Loading SALMONN-7B LoRA weights...")
    ckpt = torch.load(SALMONN_CKPT, map_location="cpu")["model"]
    prefix = "llama_model."
    lora_state = {
        k[len(prefix):]: v
        for k, v in ckpt.items()
        if k.startswith(prefix)
    }

    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    lora_loaded = [k for k in lora_state if k not in unexpected]
    print(f"LoRA keys loaded: {len(lora_loaded)} / {len(lora_state)}")
    if unexpected:
        print(f"Unexpected keys (skipped): {len(unexpected)}")

    # Merge LoRA into base weights and unwrap PEFT
    print("Merging LoRA into base weights...")
    model = model.merge_and_unload()

    # Fix generation_config before saving (temperature/top_p require do_sample=True)
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = True

    # Save merged model
    print(f"Saving merged model to {args.out} ...")
    model.save_pretrained(args.out, safe_serialization=False)

    # Also copy tokenizer
    try:
        tok = LlamaTokenizer.from_pretrained(VICUNA_PATH)
        tok.save_pretrained(args.out)
    except Exception as e:
        print(f"Tokenizer copy warning: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
