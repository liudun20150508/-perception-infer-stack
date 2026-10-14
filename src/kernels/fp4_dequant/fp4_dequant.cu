// ============================================================
// fp4_dequant.cu —— DeepSeek-V4 FP4 专家权重反量化（CUDA 版，目标 A800）
//
// 复刻 FP4(e2m1fn) 的反量化语义。这是 V4 官方 kernel.py 里
// fp4_gemm 的第一步：把 4-bit 权重按 e2m1fn 还原成浮点，再乘块缩放。
// 后续接 BF16 GEMM(cuBLAS) 即完整 W4A16 前向。
//
// 编译：nvcc -O3 -arch=sm_80 -lineinfo fp4_dequant.cu -o fp4_dequant
// ============================================================
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int FP4_BLOCK = 32;   // V4 的 fp4_block_size：每 32 个值一个 fp32 缩放

// e2m1fn：1 符号 + 2 指数 + 1 尾数（fn=有限，无 inf/nan）
__host__ __device__ inline float fp4_to_float(uint8_t x) {
    int s = (x >> 3) & 1;
    int e = (x >> 1) & 0x3;
    int m = x & 0x1;
    float mag;
    if (e == 0)      mag = 0.5f * m;          // 0 或 0.5
    else if (e == 1) mag = 1.0f + 0.5f * m;   // 1.0 或 1.5
    else if (e == 2) mag = 2.0f + 1.0f * m;   // 2.0 或 3.0
    else             mag = 4.0f + 2.0f * m;   // 4.0 或 6.0
    return (s ? -mag : mag);
}

// packed: 每字节存 2 个 fp4（低 4 位在前）；scales: 每 FP4_BLOCK 个值一个 fp32 缩放
__global__ void fp4_dequant_kernel(const uint8_t* __restrict__ packed,
                                   const float* __restrict__ scales,
                                   float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t byte = packed[i >> 1];
    uint8_t x = (i & 1) ? (byte >> 4) : (byte & 0xF);
    out[i] = fp4_to_float(x) * scales[i / FP4_BLOCK];
}

int main() {
    constexpr int N = 1 << 20;                 // 1M 个 fp4 值（约 MoE 专家权重一块的规模）
    constexpr int NBYTES = N / 2;
    constexpr int NBLOCKS = N / FP4_BLOCK;

    srand(42);
    std::vector<uint8_t> h_packed(NBYTES);
    std::vector<float> h_scales(NBLOCKS);
    for (int b = 0; b < NBYTES; ++b) h_packed[b] = (uint8_t)(rand() & 0xFF);   // 每字节 2 个随机 fp4
    for (int b = 0; b < NBLOCKS; ++b) h_scales[b] = (rand() % 1000) / 100.0f;  // 随机块缩放

    uint8_t *d_packed; float *d_scales, *d_out;
    CUDA_CHECK(cudaMalloc(&d_packed, NBYTES));
    CUDA_CHECK(cudaMalloc(&d_scales, NBLOCKS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_packed, h_packed.data(), NBYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), NBLOCKS * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256, blocks = (N + threads - 1) / threads;
    fp4_dequant_kernel<<<blocks, threads>>>(d_packed, d_scales, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    int mismatch = 0;
    for (int i = 0; i < N; ++i) {
        uint8_t byte = h_packed[i >> 1];
        uint8_t x = (i & 1) ? (byte >> 4) : (byte & 0xF);
        float expect = fp4_to_float(x) * h_scales[i / FP4_BLOCK];
        if (fabsf(h_out[i] - expect) > 1e-6f) {
            if (mismatch < 5) printf("  mismatch @%d: gpu=%f cpu=%f\n", i, h_out[i], expect);
            ++mismatch;
        }
    }

    if (mismatch == 0) {
        printf("[PASS] FP4(e2m1fn) 反量化：%d 个值 GPU==CPU 全部一致\n", N);
        printf("       样例（前 8 个）：");
        for (int i = 0; i < 8; ++i) printf("% .2f ", h_out[i]);
        printf("\n       e2m1fn 值域 = {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}\n");
    } else {
        printf("[FAIL] %d 个不一致\n", mismatch);
        return 1;
    }

    cudaFree(d_packed); cudaFree(d_scales); cudaFree(d_out);
    printf("Done. 注：这是 fp4_gemm 第一步；后续接 BF16 GEMM(cuBLAS) 即完整 W4A16 前向。\n");
    return 0;
}
