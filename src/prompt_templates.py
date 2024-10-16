PROMPT_TEMPLATES = {
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
        "instruction": "\n<|im_start|>user\n",
        "response": "\n<|im_start|>assistant\n",
    },
    "chat": {
        "instruction": "\n[|user|]\n",
        "response": "\n[|assistant|]\n",
    },
}
