#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

for benchmark in benchmark/*.yaml; do
    settings=()
    if [[ "$benchmark" == benchmark/consistory_multi.yaml ]]; then
        settings=(--seed 44 --sa_scale 0.6)
    fi
    "${PYTHON:-python}" run_eosstory.py --benchmark "$benchmark" "${settings[@]}" "$@"
done
