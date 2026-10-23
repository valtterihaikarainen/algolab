#!/usr/bin/env bash
# Download the four MNIST IDX files into data/mnist (or the directory given as $1).
# Files that already exist are skipped, so the script is safe to re-run.
set -euo pipefail

target_dir="${1:-data/mnist}"
mirror="https://ossci-datasets.s3.amazonaws.com/mnist"
files=(
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
)

mkdir -p "$target_dir"
for name in "${files[@]}"; do
  if [[ -f "$target_dir/$name" ]]; then
    echo "exists: $target_dir/$name"
    continue
  fi
  echo "downloading: $name"
  curl -fsSL "$mirror/$name.gz" | gunzip > "$target_dir/$name.tmp"
  mv "$target_dir/$name.tmp" "$target_dir/$name"
done
echo "MNIST files are in $target_dir"
