// ============================================================
// int8_gemm.cu —— INT8 量化 GEMM（量化推理主线：INT8）
//
// 三档实现对照：
//   1) naive      : 每线程一个输出元素，K 维 int 循环（baseline）
//   2) dp4a       : SIMD，一次 4 个 int8 点积（B 预转置为 [N,K]）
//   3) cuBLAS INT8: cublasGemmEx + INT8 tensor core（性能上界）
//
// 精度验证：INT8 反量化结果 vs cuBLAS FP32（量化损失）
// 自检    ：naive / dp4a / cuBLAS 三者 C_int32 逐元素应完全一致
//
// 编译：
//   nvcc -O3 -arch=sm_80 -lineinfo int8_gemm.cu -lcublas -o int8_gemm
// 运行：
//   ./int8_gemm [M] [N] [K]     # 默认 2048 2048 2048，K 需为 4 的倍数
// ============================================================
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: code=%d\n", __FILE__, __LINE__, (int)s); exit(1); } } while (0)

// ---------------- 量化 / 反量化（host，per-tensor 对称量化） ----------------
static void quantize_tensor(const float* src, int8_t* dst, size_t n, float* scale) {
    float amax = 0.0f;
    for (size_t i = 0; i < n; ++i) amax = fmaxf(amax, fabsf(src[i]));
    *scale = amax / 127.0f;
    if (*scale == 0.0f) *scale = 1.0f;          // 全零张量保护
    for (size_t i = 0; i < n; ++i) {
        float q = roundf(src[i] / *scale);
        q = fmaxf(-128.0f, fminf(127.0f, q));   // int8 范围 [-128, 127]
        dst[i] = (int8_t)q;
    }
}

// ---------------- 1) naive：每线程一个输出，K 维循环 ----------------
__global__ void int8_gemm_naive(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                int32_t* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    int acc = 0;
    for (int k = 0; k < K; ++k)
        acc += (int)A[(size_t)row * K + k] * (int)B[(size_t)k * N + col];
    C[(size_t)row * N + col] = acc;
}

// ---------------- 2) dp4a：K 步进 4，一次 4 个 int8 点积 ----------------
// B_T 是 B 的转置 [N,K]：B_T[n*K+k] = B[k*N+n]
// 这样 A 沿 K、B_T 沿 K 都是连续访存，可以直接按 int 打包喂给 __dp4a
__global__ void int8_gemm_dp4a(const int8_t* __restrict__ A, const int8_t* __restrict__ B_T,
                               int32_t* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    const int* A4 = reinterpret_cast<const int*>(A + (size_t)row * K);   // 打包 4 个 int8
    const int* B4 = reinterpret_cast<const int*>(B_T + (size_t)col * K);
    int acc = 0;
    for (int k4 = 0; k4 < K / 4; ++k4)
        acc = __dp4a(A4[k4], B4[k4], acc);   // acc += sum4(a.b)
    C[(size_t)row * N + col] = acc;
}

// ---------------- host 参考：naive int32 累加（自检 cuBLAS 转置是否写对） ----------------
static void ref_gemm_int32(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (int32_t)A[(size_t)i * K + k] * (int32_t)B[(size_t)k * N + j];
            C[(size_t)i * N + j] = acc;
        }
}

// ---------------- cuBLAS：row-major C[M,N]=A[M,K]@B[K,N] ----------------
// cuBLAS 是 column-major。推导：
//   row-major C = A@B  <=>  col-major C^T[N,M] = B^T[N,K] @ A^T[K,M]
//   内存里 B(row-major [K,N]) 被 col-major 视作 B^T，A([M,K]) 视作 A^T，均 OP_N，无需转置。
static void cublas_gemm_int8(cublasHandle_t h, const int8_t* A, const int8_t* B, int32_t* C,
                             int M, int N, int K) {
    int alpha = 1, beta = 0;
    CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
        B, CUDA_R_8I, N,     // col-major 视角 = B^T [N,K]
        A, CUDA_R_8I, K,     // col-major 视角 = A^T [K,M]
        &beta, C, CUDA_R_32I, N,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
static void cublas_gemm_fp32(cublasHandle_t h, const float* A, const float* B, float* C,
                             int M, int N, int K) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
        B, N, A, K, &beta, C, N));
}

// ---------------- 计时辅助 ----------------
template <typename F>
static double time_kernel(F launch, int iters) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i = 0; i < 5; ++i) launch();   // warmup
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
    printf("INT8 GEMM: M=%d N=%d K=%d\n", M, N, K);

    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;

    // ---- host 数据：随机 FP32 A/B，量化成 int8 ----
    std::vector<float> A_fp(nA), B_fp(nB), C_fp(nC);
    std::vector<int8_t> A_i8(nA), B_i8(nB), B_T_i8(nB);
    std::vector<int32_t> C_ref(nC);
    srand(42);
    for (size_t i = 0; i < nA; ++i) A_fp[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (size_t i = 0; i < nB; ++i) B_fp[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

    float scale_A = 0, scale_B = 0;
    quantize_tensor(A_fp.data(), A_i8.data(), nA, &scale_A);
    quantize_tensor(B_fp.data(), B_i8.data(), nB, &scale_B);
    // B 转置 [N,K]（dp4a 需要 B 沿 K 连续）
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            B_T_i8[(size_t)n * K + k] = B_i8[(size_t)k * N + n];

    // ---- device 分配 + 拷贝 ----
    int8_t *dA, *dB, *dBT; int32_t *dC; float *dAfp, *dBfp, *dCfp;
    CUDA_CHECK(cudaMalloc(&dA, nA));
    CUDA_CHECK(cudaMalloc(&dB, nB));
    CUDA_CHECK(cudaMalloc(&dBT, nB));
    CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dAfp, nA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBfp, nB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCfp, nC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, A_i8.data(), nA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B_i8.data(), nB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBT, B_T_i8.data(), nB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAfp, A_fp.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBfp, B_fp.data(), nB * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    // ---- 跑三档 INT8 GEMM + FP32 参考 ----
    std::vector<int32_t> C_naive(nC), C_dp4a(nC), C_cublas(nC);

    int8_gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaMemcpy(C_naive.data(), dC, nC * sizeof(int32_t), cudaMemcpyDeviceToHost));

    int8_gemm_dp4a<<<grid, block>>>(dA, dBT, dC, M, N, K);
    CUDA_CHECK(cudaMemcpy(C_dp4a.data(), dC, nC * sizeof(int32_t), cudaMemcpyDeviceToHost));

    cublas_gemm_int8(handle, dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaMemcpy(C_cublas.data(), dC, nC * sizeof(int32_t), cudaMemcpyDeviceToHost));

    cublas_gemm_fp32(handle, dAfp, dBfp, dCfp, M, N, K);
    CUDA_CHECK(cudaMemcpy(C_fp.data(), dCfp, nC * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 自检 1：host 参考 vs 三档（应完全一致） ----
    ref_gemm_int32(A_i8.data(), B_i8.data(), C_ref.data(), M, N, K);
    auto max_diff_i32 = [&](const std::vector<int32_t>& x, const char* name) {
        int32_t md = 0;
        for (size_t i = 0; i < nC; ++i) md = std::max(md, (int32_t)llabs((int64_t)x[i] - C_ref[i]));
        printf("  selfcheck %-8s vs host-ref: max|diff| = %d %s\n", name, md, md == 0 ? "OK" : "FAIL");
    };
    max_diff_i32(C_naive,  "naive");
    max_diff_i32(C_dp4a,   "dp4a");
    max_diff_i32(C_cublas, "cublas");

    // ---- 精度 2：INT8 反量化 vs FP32（量化损失） ----
    double deq_scale = (double)scale_A * scale_B;
    double sum_ref = 0, sum_err = 0; double max_abs = 0, max_err = 0;
    for (size_t i = 0; i < nC; ++i) {
        double dq = (double)C_cublas[i] * deq_scale;
        double ref = C_fp[i];
        double diff = fabs(dq - ref);
        sum_ref += ref * ref; sum_err += diff * diff;
        max_abs = fmax(max_abs, fabs(ref)); max_err = fmax(max_err, diff);
    }
    printf("\n[精度] INT8 vs FP32: rel_L2 = %.4e, max_abs_err = %.4e (ref_max = %.4e)\n",
           sqrt(sum_err / sum_ref), max_err, max_abs);

    // ---- 性能 ----
    int iters = 50;
    double t_naive = time_kernel([&]{ int8_gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K); }, iters);
    double t_dp4a  = time_kernel([&]{ int8_gemm_dp4a<<<grid, block>>>(dA, dBT, dC, M, N, K); }, iters);
    // cuBLAS INT8 / FP32 用 event 包住（异步）
    cudaEvent_t s, e; float ms;
    cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i = 0; i < 5; ++i) cublas_gemm_int8(handle, dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) cublas_gemm_int8(handle, dA, dB, dC, M, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e); double t_cublas_i8 = ms / iters;
    for (int i = 0; i < 5; ++i) cublas_gemm_fp32(handle, dAfp, dBfp, dCfp, M, N, K);
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) cublas_gemm_fp32(handle, dAfp, dBfp, dCfp, M, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e); double t_cublas_fp32 = ms / iters;
    cudaEventDestroy(s); cudaEventDestroy(e);

    double flops = 2.0 * M * N * K;
    printf("\n[性能] %-10s %10s %12s %10s %12s\n", "实现", "耗时(ms)", "GFLOPs", "vs naive", "vs cublasFP32");
    printf("  %-10s %10.3f %12.1f %10s %12s\n", "naive", t_naive, flops / t_naive / 1e6, "1.00x", "-");
    printf("  %-10s %10.3f %12.1f %9.2fx %12s\n", "dp4a", t_dp4a, flops / t_dp4a / 1e6, t_naive / t_dp4a, "-");
    printf("  %-10s %10.3f %12.1f %9.2fx %11.2fx\n", "cublas_i8", t_cublas_i8, flops / t_cublas_i8 / 1e6, t_naive / t_cublas_i8, t_cublas_fp32 / t_cublas_i8);
    printf("  %-10s %10.3f %12.1f %10s %12s\n", "cublas_fp32", t_cublas_fp32, flops / t_cublas_fp32 / 1e6, "-", "1.00x");

    // ---- 清理 ----
    cudaFree(dA); cudaFree(dB); cudaFree(dBT); cudaFree(dC);
    cudaFree(dAfp); cudaFree(dBfp); cudaFree(dCfp);
    cublasDestroy(handle);
    printf("\nDone.\n");
    return 0;
}
