"""
Use two docker containers to run inference and evaluation.
- https://github.com/stanleee5/text-generation-inference/tree/load_peft_safetensors
- https://github.com/stanleee5/bigcode-evaluation-harness/tree/feat/multiple

1. inference server docker 실행
2. server health check 대기
3. bigcode eval docker 실행
4. 종료 및 merged PEFT checkpoint 삭제
"""

import argparse
import os
import time

import docker
import yaml
from attrdict import AttrDict
from loguru import logger

client = docker.from_env()


def run_tgi(cfg, model_dir) -> docker.models.containers.Container:
    """
    text-generation-inference 도커 컨테이너 실
    """
    logger.warning(f"start tgi: {cfg.docker.name}")

    entrypoint = "text-generation-launcher"
    command = "--model-id /model "
    for k, v in cfg.params.items():
        command += f"--{k} {v} "

    logger.info(f"{entrypoint} {command}")
    container = client.containers.run(
        image=cfg.docker.image,
        name=cfg.docker.name,
        network=cfg.docker.network,
        entrypoint=entrypoint,
        command=command,
        auto_remove=True,
        detach=True,
        stderr=True,
        runtime="nvidia",
        shm_size="4g",
        environment=[f"NVIDIA_VISIBLE_DEVICES={cfg.docker.gpus}"],
        volumes={
            model_dir: {"bind": "/model"},
            cfg.docker.base_model_dir: {"bind": cfg.docker.base_model_dir},
            cfg.docker.cache_dir: {"bind": "/root/.cache/huggingface"},
        },
    )
    for log in container.logs(stream=True):
        log_str = log.decode("utf-8").rstrip()
        if len(log):
            logger.info(log_str)
            # print()
            if "Connected" in log_str:
                logger.warning("server connected. switch to inference")
                break
    return container


def inference_tgi(cfg, endpoint):
    """
    bigcode-evaluation-harness 도커 컨테이너 실행
    """
    logger.warning(f"start inference: {cfg.docker.name}")

    # generations/metrics 저장 경로
    save_name = f"{cfg.params.prompt}-n{cfg.params.n_samples}-max{cfg.params.max_length_generation}"
    if cfg.params.do_sample:
        save_name += f"temp{cfg.params.temperature}"

    os.makedirs(os.path.join(cfg.docker.save_root, args.save_dir), exist_ok=True)
    base_save_path = os.path.join(cfg.docker.save_root, args.save_dir, save_name)
    logger.warning(f"-> save to {base_save_path}")
    save_generation_path = base_save_path + "-generations.json"
    save_metric_path = base_save_path + "-metrics.json"
    save_log_path = base_save_path + "-logs.json"

    command = (
        "python3 main.py "
        "--save_generations "
        "--allow_code_execution "
        f"--endpoint {endpoint} "
        f"--save_generations_path {save_generation_path} "
        f"--metric_output_path {save_metric_path} "
        f"--log_output_path {save_log_path} "
    )
    for k, v in cfg.params.items():
        command += f"--{k} {v} "

    logger.info(f"{command}")
    container = client.containers.run(
        image=cfg.docker.image,
        name=cfg.docker.name,
        network=cfg.docker.network,
        working_dir=cfg.docker.working_dir,
        command=command,
        auto_remove=True,
        detach=True,
        stderr=True,
        shm_size="2g",
        volumes={
            cfg.docker.working_dir: {"bind": cfg.docker.working_dir},
            cfg.docker.save_root: {"bind": cfg.docker.save_root},
            cfg.docker.cache_dir: {"bind": "/root/.cache/huggingface"},
        },
    )
    for log in container.logs(stream=True):
        if len(log):
            print(log.decode("utf-8").rstrip())
    return container


def wait_for_tgi_server(container, wait_sec=5):
    """추론서버 로딩 대기"""
    health_check_cmd = "curl -s --head --request GET localhost:80/health"
    logger.info("Waiting server ready ...")
    while True:
        exec_res = container.exec_run(health_check_cmd)
        if "200 OK" in exec_res.output.decode("utf-8"):
            break
        time.sleep(wait_sec)
        since_time = int(time.time()) - wait_sec
        log = container.logs(since=since_time).decode("utf-8").rstrip()
        if len(log):
            print(log)
    logger.info("Server Connected ...")


def print_cfg(x, d=0):
    """recursively print dict"""
    for k, v in x.items():
        if isinstance(v, dict):
            logger.info("  " * d + k)
            print_cfg(v, d + 1)
        else:
            logger.info("  " * d + f"{k}: {v}")


def main(args):
    logger.warning(args.config_path)
    with open(args.config_path) as fileobj:
        cfg = AttrDict(yaml.safe_load(fileobj))
    print_cfg(cfg)

    endpoint = f"http://{cfg.tgi.docker.name}:{cfg.tgi.params.port}"

    tgi_container = run_tgi(cfg.tgi, args.model_dir)
    # tgi_container = client.containers.get(cfg.tgi.docker.name)

    wait_for_tgi_server(tgi_container)

    # TODO: working_dir 대신 원하는 경로에 결과 저장하도록 수정
    inference_tgi(cfg.eval, endpoint)

    # merged PEFT 파일 삭제 후 docker stop
    logger.warning(f"stop tgi: {tgi_container.name}")
    if args.remove_merged:
        # TODO: 상대경로로 확인
        if "adapter_config.json" in os.listdir(args.model_dir):
            logger.warning("Remove PEFT-merged model")
            tgi_container.exec_run("sh -c 'rm /model/model-*.safetensors'")
    tgi_container.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-path", default="configs/base.yaml")
    parser.add_argument("-m", "--model-dir", required=False)
    parser.add_argument("-s", "--save-dir", required=False)
    parser.add_argument("--remove-merged", action="store_true")
    args = parser.parse_args()

    main(args)
