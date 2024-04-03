"""
Dataset and Collator
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import yaml
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import DataCollatorForLanguageModeling
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

from src.utils import get_logger

logger = get_logger()


@dataclass
class DataArguments:
    # Load from YAML config or name/path
    data_name_or_path: str = field(default=None)
    data_config_yaml: str = field(default=None)

    instruction_key: Optional[str] = field(default="instruction")
    response_key: Optional[str] = field(default="output")
    instruction_template: str = field(default="[INST]\n")
    response_template: str = field(default="\n[/INST]\n")
    inp_strip: bool = field(default=True)
    prompt_template: str = field(default=None)  # Overwrite template

    test_size: Optional[int] = field(default=512)
    max_seq_length: int = field(default=1280)
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


def get_formatting_func(
    data_args: DataArguments,
    eos_token: str,
) -> Callable:
    def func(examples) -> str:
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
                + eos_token
            )
            texts.append(text)
        return texts

    return func


class InstructionTuningCollator(DataCollatorForCompletionOnlyLM):
    """ignore max_seq_lenth overflow warning"""

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        assert self.instruction_template, "set instruction_template for Collator"

        batch = DataCollatorForLanguageModeling.torch_call(self, examples)
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
                # warnings.warn(
                #     f"Could not find response key `{self.response_template}` in the "
                #     f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                #     f"Could not find response key `{self.response_token_ids}` in the "
                #     f'following instance: {batch["input_ids"][i]} '
                #     f"This instance will be ignored in loss calculation. "
                #     f"Note, if this happens often, consider increasing the `max_seq_length`."
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
                batch["labels"][i, :] = self.ignore_index

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

        return batch
