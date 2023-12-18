#!/bin/bash

packages=(
    "accelerate"
    "dataset"
    "deepspeed"
    "flash-attn"
    "peft"
    "transformer"
    "trl"
)

for package in "${packages[@]}"; do
    pip list | grep $package
done


# Function to extract package name
# extract_package_name() {
#     echo "$1" | sed 's/[<>=!]=.*//' | sed 's/[<>=!].*//'
# }

# # Iterate through each line in requirements.txt
# while IFS= read -r line || [[ -n "$line" ]]; do
#     # Extract the package name
#     pkg=$(extract_package_name "$line")
#     pip list | grep $pkg
# done < requirements.txt
