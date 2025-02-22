"""
Dataset and Collator
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

from src.utils import get_logger

logger = get_logger()


@dataclass
class DataArguments:
    # Load from YAML config or name/path
    data_name_or_path: str = field(default=None)
    test_size: Optional[int] = field(default=64)

    # format: {instruction, response}
    instruction_key: Optional[str] = field(default="instruction")
    response_key: Optional[str] = field(default="output")
    inp_strip: bool = field(default=True)
    prompt_template: str = field(default=None)  # Overwrite template
    instruction_template: str = field(default="[INST]\n")
    response_template: str = field(default="\n[/INST]\n")

    # format: {[conversations]}
    conversations_key: Optional[str] = field(default=None)

    dataset_num_proc: int = field(default=16)
    max_seq_length: int = field(default=2048)
    masking: Optional[bool] = field(default=True)


def load_ds(name_or_path: str) -> DatasetDict:
    if os.path.exists(name_or_path):
        return load_from_disk(name_or_path)
    else:
        return load_dataset(name_or_path)


def load_and_split_datasets(args: DataArguments) -> DatasetDict:
    data_name_or_path = args.data_name_or_path
    ds_dict = load_ds(data_name_or_path)
    if "test" not in ds_dict:
        logger.info(f"Manually split test_size: {args.test_size}")
        ds_dict = ds_dict["train"].train_test_split(test_size=args.test_size)

    logger.info(f"dataset: {ds_dict}")
    logger.info(f"len(train/test): {len(ds_dict['train'])}/{len(ds_dict['test'])}")
    return ds_dict


def get_formatting_func(data_args: DataArguments, tokenizer) -> Callable:
    eos_token_to_add = tokenizer.eos_token

    def conversations_func(examples) -> str:
        conversations = examples[data_args.conversations_key]
        texts = [
            tokenizer.apply_chat_template(conversation, tokenize=False)
            for conversation in conversations
        ]
        return texts

    def single_response_func(examples) -> str:
        texts = []
        for idx in range(len(examples[data_args.instruction_key])):
            prompt = examples[data_args.instruction_key][idx]
            if "input" in examples:
                prompt = f"{prompt}\n{examples['input'][idx]}"
            if data_args.inp_strip:
                prompt = prompt.strip()
            completion = examples[data_args.response_key][idx]
            text = (
                data_args.instruction_template
                + prompt
                + data_args.response_template
                + completion
                + eos_token_to_add
            )
            texts.append(text)
        return texts

    if data_args.conversations_key:
        logger.info("Format: Converstations with 'apply_chat_template'")
        return conversations_func
    else:
        logger.info("Format: Single-turn instruction->response")
        return single_response_func


class InstructionTuningCollator(DataCollatorForCompletionOnlyLM):
    """ignore max_seq_lenth overflow warning"""

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super(DataCollatorForCompletionOnlyLM, self).torch_call(examples)
        assert self.instruction_template

        for i in range(len(examples)):
            response_token_ids_idxs = []
            human_token_ids_idxs = []

            for assistant_idx in np.where(
                batch["labels"][i] == self.response_token_ids[0]
            )[0]:
                # find the indexes of the start of a response.
                if (
                    self.response_token_ids
                    == batch["labels"][i][
                        assistant_idx : assistant_idx + len(self.response_token_ids)
                    ].tolist()
                ):
                    response_token_ids_idxs.append(
                        assistant_idx + len(self.response_token_ids)
                    )

            if len(response_token_ids_idxs) == 0:
                # logger.info(f"{self.response_token_ids}")
                # logger.info(f"{self.tokenizer(self.response_template)}")
                # logger.info(batch["input_ids"][i].tolist())
                # logger.warning(
                #     f"Could not find response key `{self.response_template}` in the "
                #     f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                # )
                batch["labels"][i, :] = self.ignore_index

            human_token_ids = self.instruction_token_ids
            for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                # find the indexes of the start of a human answer.
                if (
                    human_token_ids
                    == batch["labels"][i][
                        human_idx : human_idx + len(human_token_ids)
                    ].tolist()
                ):
                    human_token_ids_idxs.append(human_idx)

            if len(human_token_ids_idxs) == 0:
                # logger.warning(f"{human_token_ids = }")
                # logger.warning(f"{self.tokenizer(self.instruction_template)}")
                # logger.warning(batch["input_ids"][i].tolist())
                # logger.warning(
                #     f"Could not find instruction key `{self.instruction_template}` in the "
                #     f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                # )
                batch["labels"][i, :] = self.ignore_index

            if (
                len(human_token_ids_idxs) > 0
                and len(response_token_ids_idxs) > 0
                and human_token_ids_idxs[0] > response_token_ids_idxs[0]
            ):
                human_token_ids_idxs = [0] + human_token_ids_idxs

            for idx, (start, end) in enumerate(
                zip(human_token_ids_idxs, response_token_ids_idxs)
            ):
                # Make pytorch loss function ignore all non response tokens
                if idx != 0:
                    batch["labels"][i, start:end] = self.ignore_index
                else:
                    batch["labels"][i, :end] = self.ignore_index

            if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        if self.padding_free:
            # remove padding, `attention_mask` and add `position_ids`
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = (
                attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            )
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index

        return batch
