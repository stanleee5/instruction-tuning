"""
model and tokenizer
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.utils import is_main_process

if not is_main_process():
    logger.remove()


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="JackFram/llama-160m")
    use_flash_attn: Optional[bool] = field(default=True)
    load_in_8bit: Optional[bool] = field(default=False)

    # LoRA
    use_lora: Optional[bool] = field(default=False)
    lora_r: Optional[int] = field(default=32)
    lora_alpha: Optional[int] = field(default=64)
    lora_dropout: Optional[float] = field(default=0.05)


def load_model(
    model_name_or_path: str,
    use_flash_attn: bool,
    load_in_8bit: bool,
    gradient_checkpointing: bool,
) -> torch.nn.Module:
    logger.info("Loading model")
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    if config.model_type == "llama":
        flash_params = {"use_flash_attention_2": use_flash_attn}
    elif config.model_type == "phi-msft":
        flash_params = {
            "flash_attn": use_flash_attn,
            "flash_rotary": use_flash_attn,
            "fused_dense": use_flash_attn,
        }
        # NOTE: gradient_checkpointing should be False (tarining_args)
    else:
        flash_params = {}

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_8bit=load_in_8bit,
        trust_remote_code=True,
        **flash_params,
    )

    # enable gradient checkpointing
    if gradient_checkpointing:
        logger.info("Enable gradient checkpointing")
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    logger.info(model)
    return model


def load_tokenizer(model_name_or_path: str):
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
        add_prefix_space=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.padding_side = "right"
    logger.info(tokenizer)
    return tokenizer
