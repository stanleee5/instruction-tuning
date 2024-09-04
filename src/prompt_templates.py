PROMPT_TEMPLATES = {
    "octocoder": {
        "instruction": "Question: ",
        "response": "\n\nAnswer:\n",
    },
    "wizardcoder": {
        "instruction": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n",
        "response": "\n\n### Response:\n",
    },
    "codellama": {
        "instruction": "[INST] ",
        "response": " [/INST] ",
    },
    "code": {
        "instruction": "[INST]\n",
        "response": "\n[/INST]\n",
    },
    "chatml": {
        "instruction": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n",
        "response": "<|im_end|>\n<|im_start|>assistant\n",
    },
}
