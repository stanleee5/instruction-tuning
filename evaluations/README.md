# Evaluation
Based on `text-generation-inference` and `bigcode-evaluation-harness`


### Setup
- text-generation-inference
  ```bash
  git clone https://github.com/huggingface/text-generation-inference
  cd text-generation-inference && git checkout tags/v1.4.0

  docker build -t dice/text-generation-inference:1.4.0 .
  ```

- bigcode-evaluation-harness
  ```bash
  git clone https://github.com/stanleee5/bigcode-evaluation-harness
  cd bigcode-evaluation-harness && git checkout feat/multiple

  docker build -f Dockerfile-eval -t evaluation-harness-evaluation .
  ```

### Usage

```bash
# config에서 경로/태스크 설정
python bigcode_eval.py -c configs/base.yaml
```

TODO
- [ ] directory(working_dir, save_dir) 수정
