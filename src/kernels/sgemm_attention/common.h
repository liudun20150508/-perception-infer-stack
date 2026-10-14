// common.h -- CUDA 算子公共工具
//
// 说明：本机是 Apple M1（无 CUDA），这部分代码在装有 NVIDIA GPU 的
// Linux 机器上编译运行（cd cuda && ./build.sh）。
// 国产加速器的编程模型与 CUDA 高度同构，这里展示的三种优化手法：
//   向量化访存(float4) / shared memory tiling / 在线 softmax 融合
// 迁移到 GCU 时是同一套思路，只是换成 GCU 的 runtime API 与 intrinsics。
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t e = (call);                                                 \
    if (e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(e));                                  \
      std::exit(EXIT_FAILURE);                                              \
    }                                                                       \
  } while (0)

// GELU tanh 近似的 CPU 黄金参考（双精度累加，用于校验 GPU 结果）
inline float gelu_ref(double x) {
  return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

// 计时：预热 + 多轮取最小，规避首次 launch 与时钟抖动
template <typename F>
float bench_cuda(F fn, int warmup = 3, int rounds = 20) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  float best = 1e30f;
  for (int r = 0; r < rounds; ++r) {
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    fn();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    best = std::min(best, ms);
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
  }
  return best;
}

// 与 numpy.allclose 一致的验收判据：|a-b| <= atol + rtol*|b|
inline bool allclose(const std::vector<float>& a, const std::vector<float>& b,
                     float atol = 1e-4f, float rtol = 1e-3f) {
  if (a.size() != b.size()) return false;
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double diff = std::fabs(static_cast<double>(a[i]) - b[i]);
    double tol = atol + rtol * std::fabs(static_cast<double>(b[i]));
    worst = std::max(worst, diff / tol);
  }
  std::printf("    err/tol = %.4f  (<=1.0 通过)\n", worst);
  return worst <= 1.0;
}

// 打印有效带宽与算力，供 Roofline 分析
inline void report(const char* name, float ms, double bytes, double flops) {
  std::printf("  %-26s %9.4f ms  %8.2f GB/s  %8.2f GFLOPS\n",
              name, ms, bytes / (ms * 1e-3) / 1e9, flops / (ms * 1e-3) / 1e9);
}
