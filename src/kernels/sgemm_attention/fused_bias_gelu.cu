// fused_bias_gelu.cu -- Bias + GELU 融合算子
//
// 优化点递进：
//   v1 朴素        : 每线程 1 元素，用 tanhf（SFU 上很贵）
//   v2 向量化访存  : 每线程处理 float4，一次 128bit 事务，访存合并度拉满
//   v3 快速近似    : 用 __expf 重写 GELU，避开 tanhf
//
// 为什么强调"融合"：unfused 版本需要 add kernel 写一遍中间结果、
// gelu kernel 再读一遍再写一遍，HBM 流量 4N；融合后只有 2N。
// 这个算子是访存受限的，省下的访存就是实打实的时延。
#include "common.h"

// ---- v1: 朴素实现（基线） ----
__global__ void bias_gelu_naive(const float* __restrict__ x,
                                const float* __restrict__ bias,
                                float* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i] + bias[i];
  y[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f *
          (v + 0.044715f * v * v * v)));
}

// ---- v3: float4 向量化 + __expf 快速 GELU ----
//   GELU(x) = x * sigmoid(1.5957691 * (x + 0.044715 x^3))
//   sigmoid(u) = 1 / (1 + exp(-u))
//   __expf 走 SFU，吞吐远高于 tanhf
__device__ __forceinline__ float fast_gelu(float v) {
  float u = 1.5957691216057308f * (v + 0.044715f * v * v * v);
  return v * __frcp_rn(1.0f + __expf(-u));
}

__global__ void bias_gelu_vec4(const float4* __restrict__ x,
                               const float4* __restrict__ bias,
                               float4* __restrict__ y, int n4) {
  // grid-stride loop：block 数量与问题规模解耦，便于做 occupancy 调优
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    float4 xv = x[i], bv = bias[i];
    float4 o;
    o.x = fast_gelu(xv.x + bv.x);
    o.y = fast_gelu(xv.y + bv.y);
    o.z = fast_gelu(xv.z + bv.z);
    o.w = fast_gelu(xv.w + bv.w);
    y[i] = o;
  }
}

// 尾部（n 不是 4 的倍数时）
__global__ void bias_gelu_tail(const float* __restrict__ x,
                               const float* __restrict__ bias,
                               float* __restrict__ y, int n, int start) {
  int i = start + blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = fast_gelu(x[i] + bias[i]);
}

int main() {
  const int N = 1 << 24;                       // 16M 元素 = 64MB
  const int N4 = N / 4;

  std::vector<float> hx(N), hb(N), hy(N), href(N);
  for (int i = 0; i < N; ++i) {
    hx[i] = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 6.0f;
    hb[i] = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.2f;
    href[i] = gelu_ref(static_cast<double>(hx[i]) + hb[i]);
  }

  float *dx, *db, *dy;
  CUDA_CHECK(cudaMalloc(&dx, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  double bytes = 3.0 * N * sizeof(float);      // 读 x,bias + 写 y
  double flops = 6.0 * N;                      // 粗估

  std::printf("=== CUDA: Fused Bias+GELU   N=%d (%.1f MB) ===\n",
              N, N * sizeof(float) * 3 / 1e6);

  int threads = 256;
  int blocks_naive = (N + threads - 1) / threads;

  float t1 = bench_cuda([&] {
    bias_gelu_naive<<<blocks_naive, threads>>>(dx, db, dy, N);
  });
  report("naive (1 elem/thread)", t1, bytes, flops);
  CUDA_CHECK(cudaMemcpy(hy.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("  [check] naive vs CPU ref\n");
  allclose(hy, href);

  // 向量化版本：block 数按 occupancy 给，靠 grid-stride 覆盖全量
  int blocks_vec = 4096;
  float t2 = bench_cuda([&] {
    bias_gelu_vec4<<<blocks_vec, threads>>>(
        reinterpret_cast<const float4*>(dx),
        reinterpret_cast<const float4*>(db),
        reinterpret_cast<float4*>(dy), N4);
  });
  report("float4 vec + __expf", t2, bytes, flops);
  CUDA_CHECK(cudaMemcpy(hy.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("  [check] vec4 vs CPU ref\n");
  allclose(hy, href);

  std::printf("\n  -> 向量化 + 快速近似加速比: %.2fx\n", t1 / t2);

  cudaFree(dx); cudaFree(db); cudaFree(dy);
  return 0;
}
