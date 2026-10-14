// flash_attention.cu -- FlashAttention 风格融合 Attention
//
// 为什么必须融合：naive 实现要把 scores = QK^T 这个 S×S 矩阵写进 HBM，
// 再读出来做 softmax，再读出来乘 V。
//   S=4096, B=8, H=16, fp16 时 scores 单份就是 8*16*4096*4096*2 = 4 GB
//   —— 显存直接爆炸，更别提三遍 HBM 往返的时延。
//
// Flash 的做法：按 key 分块流式扫描，块内 scores 只留在 shared memory，
// 用在线 softmax 增量修正输出，全程不写 S×S。
//   显存占用：O(S^2)  ->  O(S)
//   HBM 流量：O(S^2)  ->  O(S)
//
// 在线 softmax 更新式：
//   m_new = max(m, rowmax(s_blk))
//   l_new = l * exp(m - m_new) + rowsum(exp(s_blk - m_new))
//   acc   = acc * exp(m - m_new) + exp(s_blk - m_new) @ V_blk
#include "common.h"

#define BR 32    // query 块高（= 每 block 线程数，一线程一行）
#define BK 32    // key 块宽
#define DM 64    // head dim（上限）

// ---------------- naive：显式物化 scores，三个 kernel ----------------
__global__ void attn_scores(const float* __restrict__ Q,
                            const float* __restrict__ K,
                            float* __restrict__ S,
                            int S_len, int D, float scale) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= S_len || j >= S_len) return;
  const float* q = Q + (size_t)i * D;
  const float* k = K + (size_t)j * D;
  float acc = 0.f;
  for (int d = 0; d < D; ++d) acc += q[d] * k[d];
  S[(size_t)i * S_len + j] = acc * scale;
}

__global__ void attn_softmax(float* __restrict__ S, int S_len, bool causal) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= S_len) return;
  float* row = S + (size_t)i * S_len;
  int end = causal ? i + 1 : S_len;
  float mx = -INFINITY;
  for (int j = 0; j < end; ++j) mx = fmaxf(mx, row[j]);
  float sum = 0.f;
  for (int j = 0; j < end; ++j) { row[j] = __expf(row[j] - mx); sum += row[j]; }
  for (int j = end; j < S_len; ++j) row[j] = 0.f;
  float inv = __frcp_rn(sum);
  for (int j = 0; j < end; ++j) row[j] *= inv;
}

__global__ void attn_pv(const float* __restrict__ S,
                        const float* __restrict__ V,
                        float* __restrict__ O, int S_len, int D) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= S_len || d >= D) return;
  const float* p = S + (size_t)i * S_len;
  float acc = 0.f;
  for (int j = 0; j < S_len; ++j) acc += p[j] * V[(size_t)j * D + d];
  O[(size_t)i * D + d] = acc;
}

// ---------------- flash：分块 + 在线 softmax ----------------
__global__ void attn_flash(const float* __restrict__ Q,
                           const float* __restrict__ K,
                           const float* __restrict__ V,
                           float* __restrict__ O,
                           int S_len, int D, float scale, bool causal) {
  extern __shared__ char smem[];
  float* Qs = reinterpret_cast<float*>(smem);              // BR * D
  float* Ks = Qs + BR * DM;                                // BK * D
  float* Vs = Ks + BK * DM;                                // BK * D

  const int tid = threadIdx.x;                             // 一线程负责一行 query
  const int q0 = blockIdx.x * BR;
  const int qi = q0 + tid;

  float acc[DM];
  float m, l;
#pragma unroll
  for (int d = 0; d < DM; ++d) acc[d] = 0.f;
  m = -INFINITY;
  l = 0.f;

  // 把 Q 块搬进 shared
  if (qi < S_len) {
    for (int d = tid; d < D; d += BR) Qs[tid * DM + d] = Q[(size_t)qi * D + d];
  }

  const int jend_max = causal ? min(S_len, q0 + BR) : S_len;
  for (int j0 = 0; j0 < jend_max; j0 += BK) {
    const int jend = min(j0 + BK, jend_max);
    // 协作装载 K/V 块
    for (int idx = tid; idx < (jend - j0) * D; idx += BR) {
      int jj = idx / D, dd = idx % D;
      Ks[jj * DM + dd] = K[(size_t)(j0 + jj) * D + dd];
      Vs[jj * DM + dd] = V[(size_t)(j0 + jj) * D + dd];
    }
    __syncthreads();

    if (qi < S_len) {
#pragma unroll
      for (int jj = 0; jj < BK && (j0 + jj) < jend; ++jj) {
        int gj = j0 + jj;
        if (causal && gj > qi) continue;
        float s = 0.f;
        for (int d = 0; d < D; ++d) s += Qs[tid * DM + d] * Ks[jj * DM + d];
        s *= scale;

        // ---- 在线 softmax 更新 ----
        float mnew = fmaxf(m, s);
        float corr = __expf(m - mnew);
        float p = __expf(s - mnew);
        l = l * corr + p;
#pragma unroll
        for (int d = 0; d < DM; ++d) acc[d] = acc[d] * corr + p * Vs[jj * DM + d];
        m = mnew;
      }
    }
    __syncthreads();
  }

  if (qi < S_len) {
    float inv = __frcp_rn(l);
    for (int d = 0; d < D; ++d) O[(size_t)qi * D + d] = acc[d] * inv;
  }
}

int main() {
  const int B = 4, H = 8, S = 1024, D = 64;
  const size_t tot = static_cast<size_t>(B) * H * S * D;
  const size_t smem_sz = static_cast<size_t>(B) * H * S * S;

  std::vector<float> hQ(tot), hK(tot), hV(tot), hO(tot), hOref(tot);
  for (size_t i = 0; i < tot; ++i) {
    hQ[i] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
    hK[i] = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.2f;
    hV[i] = static_cast<float>(std::rand()) / RAND_MAX - 0.5f;
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const bool causal = true;
  const double flops = 4.0 * B * H * S * S * D;   // QK^T + PV

  float *dQ, *dK, *dV, *dO, *dS;
  CUDA_CHECK(cudaMalloc(&dQ, tot * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dK, tot * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dV, tot * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dO, tot * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dS, smem_sz * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), tot * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, hK.data(), tot * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, hV.data(), tot * sizeof(float), cudaMemcpyHostToDevice));

  std::printf("=== CUDA: Attention   B=%d H=%d S=%d D=%d causal=%d ===\n",
              B, H, S, D, static_cast<int>(causal));
  std::printf("  scores 矩阵大小: %.1f MB (naive 必须物化它)\n\n",
              smem_sz * sizeof(float) / 1e6);

  dim3 blk16(16, 16);
  dim3 gridS((S + 15) / 16, (S + 15) / 16);

  // ---- naive: 逐 (b,h) 跑三个 kernel ----
  auto naive_run = [&] {
    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < H; ++h) {
        size_t off = ((size_t)b * H + h) * S * D;
        size_t soff = ((size_t)b * H + h) * S * S;
        attn_scores<<<gridS, blk16>>>(dQ + off, dK + off, dS + soff, S, D, scale);
        attn_softmax<<<(S + 255) / 256, 256>>>(dS + soff, S, causal);
        dim3 g2((D + 15) / 16, (S + 15) / 16);
        attn_pv<<<g2, blk16>>>(dS + soff, dV + off, dO + off, S, D);
      }
    }
  };
  double naive_bytes = static_cast<double>(tot) * 4 * 4        // Q K V O
                     + static_cast<double>(smem_sz) * 4 * 3;   // scores 写+读+读
  float t1 = bench_cuda(naive_run, 2, 10);
  report("naive (物化 scores)", t1, naive_bytes, flops);
  CUDA_CHECK(cudaMemcpy(hOref.data(), dO, tot * sizeof(float), cudaMemcpyDeviceToHost));

  // ---- flash ----
  size_t shmem = (BR * DM + 2 * BK * DM) * sizeof(float);
  auto flash_run = [&] {
    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < H; ++h) {
        size_t off = ((size_t)b * H + h) * S * D;
        attn_flash<<<(S + BR - 1) / BR, BR, shmem>>>(
            dQ + off, dK + off, dV + off, dO + off, S, D, scale, causal);
      }
    }
  };
  double flash_bytes = static_cast<double>(tot) * 4 * 4;   // 只有 Q K V O
  float t2 = bench_cuda(flash_run, 2, 10);
  report("flash (在线 softmax)", t2, flash_bytes, flops);
  CUDA_CHECK(cudaMemcpy(hO.data(), dO, tot * sizeof(float), cudaMemcpyDeviceToHost));

  std::printf("\n  [check] flash vs naive\n");
  allclose(hO, hOref, 1e-3f, 1e-3f);
  std::printf("\n  -> 加速 %.2fx，HBM 流量降低 %.1fx "
              "(避免物化 %.1f MB 中间矩阵)\n",
              t1 / t2, naive_bytes / flash_bytes,
              smem_sz * sizeof(float) / 1e6);

  cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dS);
  return 0;
}
