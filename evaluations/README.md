# Evaluation
Based on `text-generation-inference` and `bigcode-evaluation-harness`


### Setup
use custom repositories

- text-generation-inference
  ```bash
  git clone https://github.com/stanleee5/text-generation-inference
  cd text-generation-inference && git checkout load_peft_safetensors

  docker build -t dice/text-generation-inference:1.3.1 .
  ```

- bigcode-evaluation-harness
  ```bash
  git clone https://github.com/stanleee5/bigcode-evaluation-harness
  cd bigcode-evaluation-harness && git checkout feat/multiple

  docker build -t dice/bigcode-evaluation-harness
  ```

### Usage

```bash
# config에서 경로/태스크 설정
python bigcode_eval.py -c configs/base.yaml
```

TODO
- [ ] directory(working_dir, save_dir) 수정
