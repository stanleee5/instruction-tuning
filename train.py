"""
Instruction tuning script
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Union, cast

import numpy as np
import torch
import transformers
from transformers import HfArgumentParser, TrainingArguments
from transformers.trainer_utils import IntervalStrategy

try:
    import wandb

    WANDB_INSTALLED = True
except ImportError:
    WANDB_INSTALLED = False

from src.data import DataArguments
from src.data import InstructionTuningCollator as ITDataCollator
from src.data import get_formatting_func, load_and_split_datasets
from src.model import ModelArguments, load_model, load_peft_config, load_tokenizer
from src.prompt_templates import PROMPT_TEMPLATES
from src.trainer import SFTTrainerModelSaveOnly as SFTTrainer
from src.utils import get_logger, is_main_process

# Setup loguru logger
logger = get_logger()


@dataclass
class TrainingArguments(TrainingArguments):
    label_names: Optional[List[str]] = field(default_factory=lambda: ["labels"])
    gradient_checkpointing: bool = field(default=True)
    eval_strategy: Union[IntervalStrategy, str] = field(default="steps")
    logging_strategy: Union[IntervalStrategy, str] = field(default="steps")
    lr_scheduler_type: str = field(default="cosine")
    remove_unused_columns: Optional[bool] = field(default=True)
    optim: str = field(default="adamw_torch")
    max_grad_norm: float = field(default=0.5)
    tf32: Optional[bool] = field(default=False)
    neftune_noise_alpha: Optional[float] = field(default=None)

    # wandb_project: Optional[str] = field(default=None)
    # wandb_name: Optional[str] = field(default=None)
    report_to: str = field(default="tensorboard")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


def parse_args():
    # Parse cli arguments and complete Args
    parser = HfArgumentParser([ModelArguments, DataArguments, TrainingArguments])
    (model_args, data_args, training_args) = parser.parse_args_into_dataclasses()

    model_args = cast(ModelArguments, model_args)
    data_args = cast(DataArguments, data_args)
    training_args = cast(TrainingArguments, training_args)
    training_args.logging_dir = training_args.output_dir

    # Overwrite instruction/response template from prompt_format
    if template := data_args.prompt_template:
        assert template in PROMPT_TEMPLATES
        data_args.instruction_template = PROMPT_TEMPLATES[template]["instruction"]
        data_args.response_template = PROMPT_TEMPLATES[template]["response"]
        logger.warning(f"Overwrite template from: {template}")
        logger.warning(
            f">> Instruction template: {repr(data_args.instruction_template)}"
        )
        logger.warning(f">> Response template: {repr(data_args.response_template)}")
    return model_args, data_args, training_args


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg
        logger.warning(f"{num_new_tokens = }")


def setup_wandb(training_args):
    if WANDB_INSTALLED:
        if "wandb" in training_args.report_to and is_main_process():
            try:
                logger.info("Using wandb for logging")
                wandb.init(
                    project=training_args.wandb_project,
                    name=training_args.wandb_name,
                    config={
                        k: v
                        for k, v in vars(training_args).items()
                        if type(v) in [float, str, int, bool, dict]
                    },
                )
            except Exception:
                logger.warning("wandb not usable")
                training_args.report_to = ["tensorboard"]
    elif "wandb" in training_args.report_to:
        logger.warning("wandb not installed")
        training_args.report_to = ["tensorboard"]


def main():
    model_args, data_args, training_args = parse_args()
    training_args.data_args = vars(data_args)
    logger.info(model_args)
    logger.info(data_args)
    logger.info(training_args)

    setup_wandb(training_args)
    set_seed(training_args.seed)

    model = load_model(training_args, model_args)
    peft_config = load_peft_config(model_args)

    tokenizer = load_tokenizer(model_args.model_name_or_path)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '[|system|]\n[EOS]\n' }}{% endif %}{{ '[|' + message['role'] + '|]\n' + message['content'] + '[EOS]\n'}}{% endfor %}{% if add_generation_prompt %}{{ '[|assistant|]\n' }}{% endif %}"
        logger.warning(f"Setting chat_template: {tokenizer.chat_template}")
        chat_special_tokens = ["[|system|]", "[|user|]", "[|assistant|]"]
        logger.info(f"Add special tokens: {chat_special_tokens}")
        smart_tokenizer_and_embedding_resize(
            {"additional_special_tokens": chat_special_tokens}, tokenizer, model
        )

    ds = load_and_split_datasets(data_args)
    formatting_func = get_formatting_func(data_args, tokenizer)
    logger.info(repr(data_args.instruction_template))
    logger.info(repr(data_args.response_template))

    data_collator = None
    if data_args.masking:
        response_template = data_args.response_template
        if isinstance(
            tokenizer,
            transformers.models.code_llama.tokenization_code_llama_fast.CodeLlamaTokenizerFast,
        ):
            # 맨 앞에 붙는 special token 제거
            logger.warning("Fixing response_template for CodeLlama")
            response_template = tokenizer.encode(
                response_template, add_special_tokens=False
            )[1:]

        data_collator = ITDataCollator(
            tokenizer=tokenizer,
            instruction_template=data_args.instruction_template,
            response_template=response_template,
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
        dataset_num_proc=data_args.dataset_num_proc,
    )
    trainer.train()


if __name__ == "__main__":
    main()
