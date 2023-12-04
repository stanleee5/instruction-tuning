"""
Dataset and Collator
"""

import warnings
from typing import Any, Dict, List, Union

import numpy as np
from transformers import DataCollatorForLanguageModeling
from trl.trainer.utils import DataCollatorForCompletionOnlyLM


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
