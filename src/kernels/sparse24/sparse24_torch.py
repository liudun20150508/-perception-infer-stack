#!/usr/bin/env python3
# ============================================================
# sparse24_torch.py —— 2:4 结构化稀疏真机加速比（A800 Sparse Tensor Core）
#
# 用 torch 底层 CUTLASS 2:4 API（torch._sparse_semi_structured_tile / _mm），
# 绕过高层 to_sparse_semi_structured（torch 2.7 封装对 cu118 有坑：_mm 未实现）。
# ============================================================
import time
import torch


def bench(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def prune_2of4(w):
    """对 [K, N] 沿 dim0 每 4 个保留 |·| 最大的 2 个（2:4 结构化剪枝）"""
    K, N = w.shape
    w4 = w.reshape(K // 4, 4, N)
    top2 = torch.topk(w4.abs(), k=2, dim=1).indices      # [K//4, 2, N]
    mask = torch.zeros_like(w4, dtype=torch.bool)
    mask.scatter_(1, top2, True)
    return (w4 * mask).reshape(K, N).contiguous()


def main():
    assert torch.cuda.is_available(), "需要 CUDA GPU（A800）"
    M = N = K = 4096
    dtype = torch.float16
    dev = "cuda"
    print(f"torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")
    print(f"shape: M=N=K={K}, dtype={dtype}")

    A = torch.randn(M, K, device=dev, dtype=dtype)
    B = torch.randn(K, N, device=dev, dtype=dtype)
    Bp = prune_2of4(B)
    print(f"2:4 剪枝后实际稀疏度 = {(Bp == 0).float().mean().item():.4f} (期望 0.5)")

    # 底层 CUTLASS 2:4 压缩（packed_t/meta_t 是 B_sp 转置的压缩）
    packed, meta, packed_t, meta_t, _ = torch._sparse_semi_structured_tile(
        Bp, algorithm="", use_cutlass=True
   )

    def sparse_mm(a):
        # A @ B_sp = (B_sp^T @ A^T)^T
        return torch._sparse_semi_structured_mm(packed_t, meta_t, a.t().contiguous()).t()

    # 正确性（vs 剪枝后稠密）
    Cd = torch.mm(A, Bp).float()
    Cs = sparse_mm(A).float()
    rel = (Cs - Cd).norm() / Cd.norm()
    mx = (Cs - Cd).abs().max()
    print(f"\n[正确性] sparse2:4 vs 剪枝后稠密: rel_L2 = {rel.item():.4e}, max_abs = {mx.item():.4e}")

    # 性能
    t_dense = bench(lambda: torch.mm(A, B))
    t_sparse = bench(lambda: sparse_mm(A))
    flops = 2 * M * N * K
    print("\n[性能]")
    print(f"  dense  FP16: {t_dense*1e3:8.3f} ms   {flops/t_dense/1e12:7.2f} TFLOPS")
    print(f"  sparse 2:4 : {t_sparse*1e3:8.3f} ms   {flops/t_sparse/1e12:7.2f} TFLOPS(按 dense 口径)")
    print(f"  ★ 加速比 = {t_dense/t_sparse:.2f}x")


if __name__ == "__main__":
    main()
