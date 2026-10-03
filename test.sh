#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
mode=single
if [[ "${1:-}" == single || "${1:-}" == multi || "${1:-}" == pose || "${1:-}" == all ]]; then
    mode="$1"
    shift
fi

case "$mode" in
    single)
        "${PYTHON:-python}" run_eosstory.py --benchmark benchmark/test.yaml "$@"
        ;;
    multi)
        "${PYTHON:-python}" run_eosstory.py --benchmark benchmark/consistory_multi.yaml \
            --max_cases 1 --seed 44 --sa_scale 0.6 "$@"
        ;;
    pose)
        "${PYTHON:-python}" run_eosstory.py --benchmark benchmark/test_controlnet.yaml "$@"
        ;;
    all)
        bash "$REPO_ROOT/test.sh" single "$@"
        bash "$REPO_ROOT/test.sh" multi "$@"
        bash "$REPO_ROOT/test.sh" pose "$@"
        ;;
esac
