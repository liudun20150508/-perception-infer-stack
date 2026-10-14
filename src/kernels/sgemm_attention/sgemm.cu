// sgemm.cu -- GEMM 优化：naive -> shared memory tiling -> float4 向量化
//
// 核心矛盾：GEMM 的算力是 O(MNK)，访存是 O(MK+KN+MN)。
// 不复用数据的话，取数/计算比是 O(1)，直接卡在带宽上。
// Shared Memory Tiling 把取数/计算比提到 O(TILE)，
// 这是把 GEMM 从访存受限推向算力受限的唯一手段。
//
//  arithmetic intensity ≈ 2*M*N*K / (4*(MK+KN+MN))
//  未分块时每读一个 A/B 元素只做 1 次 FMA；分块 TILE=32 后，
//  每个从 global 读进来的元素被复用 32 次。
#include "common.h"

#define TILE 32

// ---- v1: 朴素，每线程一个输出元素，全程直接读 global ----
__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];   // B 跨步访问，未合并
  }
  C[row * N + col] = acc;
}

// ---- v2: shared memory 分块 ----
__global__ void gemm_tiled(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.f;

  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int kt = t * TILE + threadIdx.x;
    As[threadIdx.y][threadIdx.x] =
        (row < M && kt < K) ? A[row * K + kt] : 0.f;
    kt = t * TILE + threadIdx.y;
    Bs[threadIdx.y][threadIdx.x] =
        (kt < K && col < N) ? B[kt * N + col] : 0.f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];   // 全走 shared
    }
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

// ---- v3: 分块 + float4 向量化装载 ----
// 每个线程一次搬 4 个 float，把 4 次 32bit 事务合成 1 次 128bit 事务，
// 直接把装载指令数降到 1/4，缓解分块 kernel 里的取数瓶颈。
__global__ void gemm_tiled_vec4(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.f;

  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int base = t * TILE;
    // 行方向用 float4 连续装载 A
    if ((base + threadIdx.x * 4) < K && row < M &&
        (threadIdx.x * 4) < TILE) {
      const float4 a = *reinterpret_cast<const float4*>(
          &A[row * K + base + threadIdx.x * 4]);
      As[threadIdx.y][threadIdx.x * 4 + 0] = a.x;
      As[threadIdx.y][threadIdx.x * 4 + 1] = a.y;
      As[threadIdx.y][threadIdx.x * 4 + 2] = a.z;
      As[threadIdx.y][threadIdx.x * 4 + 3] = a.w;
    }
    if ((base + threadIdx.y * 4) < K && col < N &&
        (threadIdx.y * 4) < TILE) {
      const float4 b = *reinterpret_cast<const float4*>(
          &B[(base + threadIdx.y * 4) * N + col]);
      Bs[threadIdx.y * 4 + 0][threadIdx.x] = b.x;
      Bs[threadIdx.y * 4 + 1][threadIdx.x] = b.y;
      Bs[threadIdx.y * 4 + 2][threadIdx.x] = b.z;
      Bs[threadIdx.y * 4 + 3][threadIdx.x] = b.w;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

int main() {
  const int M = 2048, N = 2048, K = 2048;
  const size_t sa = static_cast<size_t>(M) * K * sizeof(float);
  const size_t sb = static_cast<size_t>(K) * N * sizeof(float);
  const size_t sc = static_cast<size_t>(M) * N * sizeof(float);

  std::vector<float> hA(M * K), hB(K * N), hC(M * N, 0.f);
  for (auto& v : hA) v = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
  for (auto& v : hB) v = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, sa));
  CUDA_CHECK(cudaMalloc(&dB, sb));
  CUDA_CHECK(cudaMalloc(&dC, sc));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), sa, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), sb, cudaMemcpyHostToDevice));

  double flops = 2.0 * M * N * K;
  double bytes = static_cast<double>(sa + sb + sc);
  std::printf("=== CUDA: SGEMM   M=%d N=%d K=%d ===\n", M, N, K);
  std::printf("  算术强度 AI = %.1f FLOP/Byte "
              "(A100 ridge point ≈ 12.5，越高越算力受限)\n\n",
              flops / bytes);

  dim3 blk(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  float t1 = bench_cuda([&] { gemm_naive<<<grid, blk>>>(dA, dB, dC, M, N, K); });
  report("naive (global only)", t1, bytes, flops);

  float t2 = bench_cuda([&] { gemm_tiled<<<grid, blk>>>(dA, dB, dC, M, N, K); });
  report("shared memory tiling", t2, bytes, flops);

  float t3 = bench_cuda([&] { gemm_tiled_vec4<<<grid, blk>>>(dA, dB, dC, M, N, K); });
  report("tiling + float4", t3, bytes, flops);

  std::printf("\n  -> tiling 加速 %.2fx，再叠加 float4 后共 %.2fx\n",
              t1 / t2, t1 / t3);

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}
