#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
g++ -O2 -std=c++17 -pthread dag_exec.cpp -o dag_exec
./dag_exec
