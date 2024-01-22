"""
Instruction tuning script
"""

import os
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Union, cast

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_from_disk
from loguru import logger
from transformers import HfArgumentParser, TrainingArguments
from transformers.trainer_utils import IntervalStrategy

from src.data import InstructionTuningCollator as ITDataCollator
from src.model import ModelArguments, load_model, load_peft_config, load_tokenizer
from src.trainer import SFTTrainerNoDeepspeedSave as SFTTrainer
from src.utils import get_logger, setup_loguru_logging_intercept

# Setup loguru logger
logger = get_logger()
setup_loguru_logging_intercept()
datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_info()


@dataclass
class TrainingArguments(TrainingArguments):
    # TODO: default cannot be ["labels"]. Pass it to cli hparam
    # label_names: Optional[List[str]] = field(default=["labels"])
    gradient_checkpointing: bool = field(default=True)
    evaluation_strategy: Union[IntervalStrategy, str] = field(default="steps")
    logging_strategy: Union[IntervalStrategy, str] = field(default="steps")
    lr_scheduler_type: str = field(default="cosine")
    remove_unused_columns: Optional[bool] = field(default=False)
    optim: str = field(default="adamw_torch")
    tf32: Optional[bool] = field(default=True)
    neftune_noise_alpha: Optional[float] = field(default=None)


@dataclass
class DataArguments:
    data_name_or_path: str = field(default=None)

    instruction_key: Optional[str] = field(default="instruction")
    response_key: Optional[str] = field(default="output")
    instruction_template: str = field(default="[INST]\n")
    response_template: str = field(default="\n[/INST]\n")
    inp_strip: bool = field(default=True)

    test_size: Optional[int] = field(default=512)
    max_seq_length: int = field(default=1536)
    masking: Optional[bool] = field(default=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


def load_and_split_dataset(data_args: DataArguments):
    data_name_or_path = data_args.data_name_or_path
    load_fn = load_from_disk if os.path.exists(data_name_or_path) else load_dataset
    ds = load_fn(data_name_or_path)

    if "test" not in ds:
        logger.info(f"Manually split test_size: {data_args.test_size}")
        ds = ds["train"].train_test_split(test_size=data_args.test_size)

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
        data_collator = ITDataCollator(
            tokenizer=tokenizer,
            instruction_template=data_args.instruction_template,
            response_template=response_token_ids[1:],
        )

    model = load_model(training_args, model_args)
    peft_config = load_peft_config(model_args)

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
