"""
model and tokenizer
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        use_flash_attention_2=use_flash_attn,
        load_in_8bit=load_in_8bit,
        trust_remote_code=True,
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
        model_name_or_path, padding_side="right", add_prefix_space=False
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.padding_side = "right"
    logger.info(tokenizer)
    return tokenizer
