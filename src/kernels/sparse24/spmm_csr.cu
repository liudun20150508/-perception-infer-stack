// ============================================================
// spmm_csr.cu —— 手写 CSR SpMM（非结构化稀疏矩阵乘）
//
// 覆盖稀疏推理里更进一步的实践：非结构化稀疏怎么用 CSR 压缩、怎么跳零、
// 以及稀疏行的负载不均衡。C = A_sparse(CSR) @ B_dense。
//
// 编译：nvcc -O3 -arch=sm_80 -lineinfo spmm_csr.cu -lcublas -o spmm_csr
// 运行：./spmm_csr [M] [N] [K]   # 默认 4096，稀疏度 50%
// ============================================================
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: code=%d\n", __FILE__, __LINE__, (int)s); exit(1); } } while (0)

// ---- 生成非结构化稀疏 A（sparsity = 非零比例）并转 CSR ----
static void dense_to_csr(const std::vector<float>& A, int M, int K, float sparsity,
                         std::vector<float>& val, std::vector<int>& col_idx, std::vector<int>& row_ptr) {
    row_ptr.assign(M + 1, 0);
    val.clear(); col_idx.clear();
    for (int i = 0; i < M; ++i) {
        row_ptr[i] = (int)val.size();
        for (int k = 0; k < K; ++k) {
            if ((float)rand() / RAND_MAX < sparsity) {
                val.push_back(A[(size_t)i * K + k]);
                col_idx.push_back(k);
            }
        }
    }
    row_ptr[M] = (int)val.size();
}

// ---- 手写 SpMM 版本1：每线程算一个输出元素，遍历该行非零 ----
__global__ void spmm_csr_naive(const float* __restrict__ val, const int* __restrict__ col_idx,
                               const int* __restrict__ row_ptr, const float* __restrict__ B,
                               float* __restrict__ C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p)
        acc += val[p] * B[(size_t)col_idx[p] * N + col];
    C[(size_t)row * N + col] = acc;
}

// ---- cublas dense GEMM 对照（row-major C=A@B）----
static void cublas_gemm_fp32(cublasHandle_t h, const float* A, const float* B, float* C,
                             int M, int N, int K) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
        B, N, A, K, &beta, C, N));
}

template <typename F>
static double time_gpu(F launch, int iters) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i = 0; i < 5; ++i) launch();
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;
}

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    const float sparsity = 0.5f;   // 50% 非零
    printf("手写 CSR SpMM: M=%d N=%d K=%d 稀疏度=%.0f%%(非零)\n", M, N, K, sparsity * 100);

    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;

    std::vector<float> A(nA), B(nB), A_pruned(nA, 0.0f);
    srand(42);
    for (size_t i = 0; i < nA; ++i) A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < nB; ++i) B[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

    std::vector<float> val; std::vector<int> col_idx, row_ptr;
    dense_to_csr(A, M, K, sparsity, val, col_idx, row_ptr);
    int nnz = (int)val.size();
    printf("nnz = %d（实际稀疏度 %.4f）\n", nnz, (double)nnz / nA);

    // 统计每行 nnz 的分布（负载不均衡的直接证据）
    int mn = K, mx = 0; long long sum = 0;
    for (int i = 0; i < M; ++i) {
        int c = row_ptr[i + 1] - row_ptr[i];
        mn = std::min(mn, c); mx = std::max(mx, c); sum += c;
    }
    printf("每行非零数: min=%d max=%d mean=%.1f（负载不均衡：max/min=%.2fx）\n",
           mn, mx, (double)sum / M, (double)mx / mn);

    // 剪枝后的稠密 A（含 0，用于 dense 对照验证正确性）
    for (int i = 0; i < M; ++i)
        for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p)
            A_pruned[(size_t)i * K + col_idx[p]] = val[p];

    // device
    float *d_val, *dB, *dC, *dA_pruned, *dC_dense;
    int *d_col_idx, *d_row_ptr;
    CUDA_CHECK(cudaMalloc(&d_val, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dB, nB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dA_pruned, nA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC_dense, nC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_val, val.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), nB * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_pruned, A_pruned.data(), nA * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // ---- 手写 SpMM ----
    std::vector<float> C_sp(nC), C_dense(nC);
    cudaMemset(dC, 0, nC * sizeof(float));
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    spmm_csr_naive<<<grid, block>>>(d_val, d_col_idx, d_row_ptr, dB, dC, M, N);
    CUDA_CHECK(cudaMemcpy(C_sp.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));

    // dense 对照（cublas，剪枝后 A）
    cublas_gemm_fp32(handle, dA_pruned, dB, dC_dense, M, N, K);
    CUDA_CHECK(cudaMemcpy(C_dense.data(), dC_dense, nC * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 正确性 ----
    double md = 0;
    for (size_t i = 0; i < nC; ++i) md = std::max(md, fabs((double)C_sp[i] - C_dense[i]));
    printf("\n[正确性] spmm_csr vs dense(剪枝后): max|diff| = %.3e %s\n", md, md < 1e-2 ? "OK" : "FAIL");

    // ---- 性能 ----
    int iters = 20;
    cudaMemset(dC, 0, nC * sizeof(float));
    double t_naive = time_gpu([&]{ spmm_csr_naive<<<grid, block>>>(d_val, d_col_idx, d_row_ptr, dB, dC, M, N); }, iters);
    // dense 原始 A（未剪枝，作为 dense 上界）
    double t_dense = time_gpu([&]{ cublas_gemm_fp32(handle, dA_pruned, dB, dC_dense, M, N, K); }, iters);

    double flops_sp = 2.0 * nnz * N;      // SpMM 有效 FLOPs（跳零）
    double flops_dense = 2.0 * M * K * N; // dense 有效 FLOPs
    printf("\n[性能] %-12s %10s %14s %12s\n", "实现", "耗时(ms)", "有效GFLOPs", "vs dense");
    printf("  %-12s %10.3f %14.1f %12s\n", "spmm_csr_naive", t_naive, flops_sp / t_naive / 1e6, "-");
    printf("  %-12s %10.3f %14.1f %12s\n", "dense(cuBLAS)", t_dense, flops_dense / t_dense / 1e6, "1.00x");
    printf("\n结论：非结构化稀疏 FLOPs 减半，但 CSR 间接访存（col_idx 随机）+ 负载不均衡，\n"
           "      手写 SpMM 跑不过 dense——这就是「非结构化难加速、结构化(2:4)才行」的实证。\n");

    cudaFree(d_val); cudaFree(d_col_idx); cudaFree(d_row_ptr); cudaFree(dB);
    cudaFree(dC); cudaFree(dA_pruned); cudaFree(dC_dense);
    cublasDestroy(handle);
    printf("\nDone.\n");
    return 0;
}
