"""
Instruction tuning script
"""

import os
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union, cast

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_from_disk
from loguru import logger
from peft.tuners.lora import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.trainer_utils import IntervalStrategy

from src.data import InstructionTuningCollator as DataCollator
from src.trainer import SFTTrainerNoDeepspeedSave as SFTTrainer
from src.utils import is_main_process, setup_loguru_logging_intercept

# Setup loguru logger
if not is_main_process():
    logger.remove()
else:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

setup_loguru_logging_intercept(
    modules=("accelerate", "datasets", "transformers", "DeepSpeed")
)


@dataclass
class TrainingArguments(TrainingArguments):
    label_names: Optional[List[str]] = field(default="labels")
    gradient_checkpointing: bool = field(default=True)
    evaluation_strategy: Union[IntervalStrategy, str] = field(default="steps")
    logging_strategy: Union[IntervalStrategy, str] = field(default="steps")
    lr_scheduler_type: str = field(default="cosine")
    remove_unused_columns: Optional[bool] = field(default=False)
    optim: str = field(default="adamw_torch")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="robowaifudev/megatron-gpt2-345m")
    use_flash_attn: Optional[bool] = field(default=True)
    load_in_8bit: Optional[bool] = field(default=False)
    # LoRA
    use_lora: Optional[bool] = field(default=False)
    lora_r: Optional[int] = field(default=32)
    lora_alpha: Optional[int] = field(default=64)
    lora_dropout: Optional[float] = field(default=0.1)


@dataclass
class DataArguments:
    data_name_or_path: str = field(default=None)

    instruction_key: Optional[str] = field(default="instruction")
    response_key: Optional[str] = field(default="output")
    instruction_template: str = field(default="[INST]\n")
    response_template: str = field(default="\n[/INST]\n")
    inp_strip: bool = field(default=True)

    test_size: Optional[int] = field(default=None)
    test_ratio: Optional[float] = field(default=0.1)
    max_seq_length: int = field(default=1536)
    masking: Optional[bool] = field(default=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


def load_model(
    model_name_or_path: str,
    use_flash_attn: bool,
    load_in_8bit: bool,
    gradient_checkpointing: bool,
):
    logger.info("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        use_flash_attention_2=use_flash_attn,
        load_in_8bit=load_in_8bit,
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
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.padding_side = "right"
    logger.info(tokenizer)
    return tokenizer


def load_and_split_dataset(data_args: DataArguments):
    data_name_or_path = data_args.data_name_or_path
    load_fn = load_from_disk if os.path.exists(data_name_or_path) else load_dataset
    ds = load_fn(data_name_or_path)

    if "test" not in ds:
        test_split = (
            data_args.test_size if data_args.test_size else data_args.test_ratio
        )
        logger.info(f"Manually split test_split: {test_split}")
        ds = ds["train"].train_test_split(test_size=test_split)

    logger.info(f"dataset: {ds}")
    logger.info(f"len(dataset): {len(ds['train'])}/{len(ds['test'])}")
    return ds


def get_formatting_func(
    instruction_template: str,
    response_template: str,
    instruction_key: str,
    response_key: str,
    inp_strip: bool,
) -> Callable:
    def func(examples) -> str:
        texts = []
        for idx in range(len(examples[instruction_key])):
            prompt = examples[instruction_key][idx]
            if "input" in examples:
                prompt = f"{prompt}\n{examples['input'][idx]}"
            completion = examples[response_key][idx]
            if inp_strip:
                prompt = prompt.strip()
            texts.append(instruction_template + prompt + response_template + completion)
        return texts

    return func


def main():
    parser = HfArgumentParser([ModelArguments, DataArguments, TrainingArguments])
    (model_args, data_args, training_args) = parser.parse_args_into_dataclasses()
    model_args = cast(ModelArguments, model_args)
    data_args = cast(DataArguments, data_args)
    training_args = cast(TrainingArguments, training_args)
    training_args.logging_dir = training_args.output_dir

    logger.info(training_args)
    logger.info(model_args)
    logger.info(data_args)

    set_seed(training_args.seed)
    tokenizer = load_tokenizer(model_args.model_name_or_path)
    ds = load_and_split_dataset(data_args)

    # set formatting function
    formatting_func = get_formatting_func(
        data_args.instruction_template,
        data_args.response_template,
        data_args.instruction_key,
        data_args.response_key,
        data_args.inp_strip,
    )

    data_collator = None
    if data_args.masking:
        response_token_ids = tokenizer.encode(
            data_args.response_template, add_special_tokens=False
        )
        # Ignore first space token for CodeLlama tokenizer
        # TODO: check with other tokenizers
        data_collator = DataCollator(
            tokenizer=tokenizer,
            instruction_template=data_args.instruction_template,
            response_template=response_token_ids[1:],
        )

    peft_config = None
    if model_args.use_lora:
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )

    model = load_model(
        model_args.model_name_or_path,
        model_args.use_flash_attn,
        model_args.load_in_8bit,
        training_args.gradient_checkpointing,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        peft_config=peft_config,
        formatting_func=formatting_func,
        max_seq_length=data_args.max_seq_length,
    )
    trainer.train()


if __name__ == "__main__":
    main()
