"""
model and tokenizer
"""

import ast
from dataclasses import dataclass, field
from typing import Optional

import torch
from loguru import logger
from peft.tuners.lora import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.utils import get_logger

logger = get_logger()


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="JackFram/llama-160m")
    use_flash_attn: Optional[bool] = field(default=True)
    load_in_8bit: Optional[bool] = field(default=False)

    # LoRA
    use_lora: Optional[bool] = field(default=False)
    lora_r: Optional[int] = field(default=32)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)
    # convert to list using ast.literal_eval (e.g. ["k_proj"])
    target_modules: Optional[str] = field(default="None")


def load_peft_config(model_args: ModelArguments):
    peft_config = None
    if model_args.use_lora:
        target_modules = ast.literal_eval(model_args.target_modules)
        logger.info(f"{target_modules = }")

        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
        )
    return peft_config


def load_model(training_args, model_args) -> torch.nn.Module:
    logger.info("Loading model")
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        use_flash_attention_2=model_args.use_flash_attn,
        load_in_8bit=model_args.load_in_8bit,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # enable gradient checkpointing
    if training_args.gradient_checkpointing:
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
        trust_remote_code=True,
        add_eos_token=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.unk_token})
    logger.info(tokenizer)
    return tokenizer
