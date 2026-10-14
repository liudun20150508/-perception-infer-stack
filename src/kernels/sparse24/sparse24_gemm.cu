// ============================================================
// sparse24_gemm.cu —— 2:4 结构化稀疏 GEMM（结构化稀疏实践）
//
// 1) 对权重 B[K,N] 沿 K 维做 2:4 剪枝：每 4 个连续元素保留 |·| 最大的 2 个
// 2) 压缩：每 4 元素块 → 2 个非零 value + 4-bit 掩码（两个 2-bit 索引）
// 3) 手写跳过零元 kernel（有效 FLOPs 减半） vs dense naive kernel
// 4) 精度：稀疏 GEMM vs cuBLAS FP32（原始 B）→ 剪枝损失
//
// 编译：nvcc -O3 -arch=sm_80 -lineinfo sparse24_gemm.cu -lcublas -o sparse24_gemm
// 运行：./sparse24_gemm [M] [N] [K]     # K 需为 4 的倍数
// ============================================================
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: code=%d\n", __FILE__, __LINE__, (int)s); exit(1); } } while (0)

// ---------------- 2:4 剪枝（host）：B[K,N] 沿 K 维每 4 个保留 top2 ----------------
static void prune_2of4_host(const float* B, float* B_pruned, float* B_val, uint8_t* B_idx,
                            int K, int N) {
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; k += 4) {
            float v[4];
            for (int j = 0; j < 4; ++j) v[j] = B[(size_t)(k + j) * N + n];
            // 找 |·| 最大的两个索引
            int m0 = 0;
            for (int j = 1; j < 4; ++j) if (fabsf(v[j]) > fabsf(v[m0])) m0 = j;
            int m1 = (m0 == 0) ? 1 : 0;
            for (int j = 0; j < 4; ++j) if (j != m0 && fabsf(v[j]) > fabsf(v[m1])) m1 = j;
            int i0 = (m0 < m1) ? m0 : m1;   // 保证 idx0 < idx1
            int i1 = (m0 < m1) ? m1 : m0;
            // 写回剪枝后的稠密矩阵（非零位保留，其余 0）
            for (int j = 0; j < 4; ++j)
                B_pruned[(size_t)(k + j) * N + n] = (j == i0 || j == i1) ? v[j] : 0.0f;
            // 压缩：每块 2 个 value + 4-bit 掩码
            int blk = k / 4;
            B_val[(size_t)n * (K / 2) + (size_t)blk * 2 + 0] = v[i0];
            B_val[(size_t)n * (K / 2) + (size_t)blk * 2 + 1] = v[i1];
            B_idx[(size_t)n * (K / 4) + blk] = (uint8_t)(i0 | (i1 << 2));
        }
    }
}

// ---------------- dense naive：遍历所有 K（含 0 元） ----------------
__global__ void dense_gemm_naive(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    C[(size_t)row * N + col] = acc;
}

// ---------------- sparse 2:4：只累加每个块里的 2 个非零元（FLOPs 减半） ----------------
__global__ void sparse24_gemm(const float* __restrict__ A, const float* __restrict__ B_val,
                              const uint8_t* __restrict__ B_idx, float* __restrict__ C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    const int nblocks = K / 4;
    const float* Aval = A + (size_t)row * K;
    const float* Bv   = B_val + (size_t)col * (K / 2);
    const uint8_t* Bi = B_idx + (size_t)col * nblocks;
    float acc = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        uint8_t m = Bi[b];
        int idx0 = m & 3, idx1 = (m >> 2) & 3;
        float a0 = Aval[(size_t)b * 4 + idx0];
        float a1 = Aval[(size_t)b * 4 + idx1];
        acc += a0 * Bv[(size_t)b * 2 + 0] + a1 * Bv[(size_t)b * 2 + 1];
    }
    C[(size_t)row * N + col] = acc;
}

// ---------------- cuBLAS FP32 参考：row-major C[M,N]=A[M,K]@B[K,N] ----------------
static void cublas_gemm_fp32(cublasHandle_t h, const float* A, const float* B, float* C,
                             int M, int N, int K) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
        B, N, A, K, &beta, C, N));
}

// 计时 kernel（cudaEvent）
template <typename F>
static double time_gpu(F launch, int iters) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i = 0; i < 5; ++i) launch();
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;
}

int main(int argc, char** argv) {
    int M = 2048, N = 2048, K = 2048;
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (K % 4 != 0) { fprintf(stderr, "K must be multiple of 4 (got %d)\n", K); return 1; }
    printf("2:4 结构化稀疏 GEMM: M=%d N=%d K=%d\n", M, N, K);

    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;

    std::vector<float> A(nA), B(nB), B_pruned(nB), C_ref(nC);
    std::vector<float> B_val((size_t)N * (K / 2));
    std::vector<uint8_t> B_idx((size_t)N * (K / 4));
    srand(42);
    for (size_t i = 0; i < nA; ++i) A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < nB; ++i) B[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

    prune_2of4_host(B.data(), B_pruned.data(), B_val.data(), B_idx.data(), K, N);
    // 实际稀疏度（应为 0.5）
    size_t zeros = 0;
    for (size_t i = 0; i < nB; ++i) if (B_pruned[i] == 0.0f) ++zeros;
    printf("  实际稀疏度 = %.4f (期望 0.5000)\n", (double)zeros / nB);

    float *dA, *dB, *dBp, *dBval, *dC, *dCref;
    uint8_t *dBidx;
    CUDA_CHECK(cudaMalloc(&dA, nA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, nB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBp, nB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBval, B_val.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBidx, B_idx.size()));
    CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCref, nC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), nB * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBp, B_pruned.data(), nB * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBval, B_val.data(), B_val.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBidx, B_idx.data(), B_idx.size(), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    std::vector<float> C_dense(nC), C_sparse(nC);

    // dense（剪枝后的稠密，含 0）
    dense_gemm_naive<<<grid, block>>>(dA, dBp, dC, M, N, K);
    CUDA_CHECK(cudaMemcpy(C_dense.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
    // sparse 2:4
    sparse24_gemm<<<grid, block>>>(dA, dBval, dBidx, dC, M, N, K);
    CUDA_CHECK(cudaMemcpy(C_sparse.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
    // 参考：cuBLAS FP32（原始 B，未剪枝）
    cublas_gemm_fp32(handle, dA, dB, dCref, M, N, K);
    CUDA_CHECK(cudaMemcpy(C_ref.data(), dCref, nC * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 精度 1：sparse vs dense（剪枝后稠密）——验证 sparse kernel 正确性 ----
    double md = 0;
    for (size_t i = 0; i < nC; ++i) md = fmax(md, fabs((double)C_sparse[i] - C_dense[i]));
    printf("\n[正确性] sparse2:4 vs dense(剪枝后): max|diff| = %.3e %s\n",
           md, md < 1e-3 ? "OK" : "FAIL");

    // ---- 精度 2：剪枝损失（sparse vs 原始 B 的 FP32 参考） ----
    double sum_ref = 0, sum_err = 0, max_abs = 0, max_err = 0;
    for (size_t i = 0; i < nC; ++i) {
        double d = fabs((double)C_sparse[i] - C_ref[i]);
        sum_ref += (double)C_ref[i] * C_ref[i]; sum_err += d * d;
        max_abs = fmax(max_abs, fabs((double)C_ref[i])); max_err = fmax(max_err, d);
    }
    printf("[剪枝损失] rel_L2 = %.4e, max_abs_err = %.4e (ref_max = %.4e)\n",
           sqrt(sum_err / sum_ref), max_err, max_abs);

    // ---- 性能 ----
    int iters = 50;
    double t_dense = time_gpu([&]{ dense_gemm_naive<<<grid, block>>>(dA, dBp, dC, M, N, K); }, iters);
    double t_sparse = time_gpu([&]{ sparse24_gemm<<<grid, block>>>(dA, dBval, dBidx, dC, M, N, K); }, iters);
    double flops_d = 2.0 * M * N * K;          // dense 有效 FLOPs
    double flops_s = (double)M * N * K;        // sparse 有效 FLOPs（跳一半）
    printf("\n[性能] %-8s %10s %12s %12s\n", "实现", "耗时(ms)", "GFLOPs(dense口径)", "有效GFLOPs");
    printf("  %-8s %10.3f %12.1f %12s\n", "dense", t_dense, flops_d / t_dense / 1e6, "-");
    printf("  %-8s %10.3f %12.1f %12.1f\n", "sparse2:4", t_sparse, flops_d / t_sparse / 1e6, flops_s / t_sparse / 1e6);
    printf("  墙钟比 sparse/dense = %.3fx（手写跳零不一定更快：访存不规则，真加速靠 Sparse Tensor Core）\n",
           t_dense / t_sparse);

    cudaFree(dA); cudaFree(dB); cudaFree(dBp); cudaFree(dBval); cudaFree(dBidx);
    cudaFree(dC); cudaFree(dCref);
    cublasDestroy(handle);
    printf("\nDone.\n");
    return 0;
}
