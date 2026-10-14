#!/usr/bin/env bash
# OS 层证据：拓扑 + 亲和性/伪共享/NUMA 实测
set -uo pipefail
cd "$(dirname "$0")"

echo "---- 系统拓扑 ----"
lscpu 2>/dev/null | grep -iE "^CPU\(s\)|Model name|Thread\(s\)|Core\(s\)|Socket\(s\)|NUMA node" || true
echo "  内核: $(uname -r)"
echo "  cgroup 负载: $(cat /proc/loadavg)"
echo

g++ -O2 -std=c++17 -pthread os_bench.cpp -o os_bench || { echo "编译失败"; exit 1; }
./os_bench "${1:-20000000}"
