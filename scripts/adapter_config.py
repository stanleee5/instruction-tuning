"""
adapter_config.json 파일의 base_model_name_or_path 업데이트
"""
import argparse
import json
import os


def update_json_file(file_path, old_value, new_value):
    with open(file_path, "r") as file:
        data = json.load(file)
        if data.get("base_model_name_or_path") == old_value:
            data["base_model_name_or_path"] = new_value
            print(file_path)
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)


def update_all_json_files_in_directory(root_path, old_value, new_value):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file == "adapter_config.json":
                file_path = os.path.join(root, file)
                update_json_file(file_path, old_value, new_value)


if __name__ == "__main__":
    # 경로, 이전 값, 새 값 설정
    # root_directory / old_base_model_path / new_base_model_path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", required=True, help="trained ckpt dir", default="/trained"
    )
    parser.add_argument(
        "--old",
        required=True,
        default="/opt/ml/input/data/loaded_model",
        help="example for sagemaker",
    )
    parser.add_argument(
        "--new",
        required=True,
        help="local base model path",
        default="codellama/CodeLlama-7b-hf",
    )
    args = parser.parse_args()

    # JSON 파일 업데이트 실행
    update_all_json_files_in_directory(args.root, args.old, args.new)
