import argparse
import os

import torch
from loguru import logger
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def download_and_unload_peft(model_dir, dtype, revision, trust_remote_code):
    torch_dtype = torch.float16
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    logger.info("Peft model detected.")
    logger.info("Loading the model it might take a while without feedback")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        revision=revision,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    logger.info(f"Loaded.")
    logger.info(f"Merging the lora weights.")

    base_model_dir = model.peft_config["default"].base_model_name_or_path

    model = model.merge_and_unload()

    os.makedirs(model_dir, exist_ok=True)
    cache_dir = model_dir
    logger.info(f"Saving the newly created merged model to {cache_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.save_pretrained(cache_dir)

    model.save_pretrained(cache_dir, safe_serialization=True)
    model.config.save_pretrained(cache_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()

    download_and_unload_peft(args.model, args.dtype, None, True)
