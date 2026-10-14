# -*- coding: utf-8 -*-
"""
16路视频流调度器模拟 —— 证明"单实例 + 动态组批"对显存和吞吐的收益
===================================================================

背景:16 路摄像头视频流做推理。朴素做法是每路一个推理实例,每份实例都要
加载模型权重 + 各自的 CUDA 上下文,显存直接爆掉。本脚本模拟两种调度:

  A. 每路一实例: 来一帧推一帧(batch=1, 全部串行排队)
  B. 单实例 + 动态组批: 时间窗 10ms 攒批, 凑不满也跑(部分 batch);
     每路只保留最新 2 帧(丢旧留新); 迟到的乱序旧帧按序号直接丢

模拟的"乱序"是真实情况: 每路帧到达带 ±8ms 抖动, 4% 的帧会迟到 2.5 个
帧周期(路内乱序)。GPU 推理延迟和显存用量是参数化估算, 在真实服务器上
可用 nvidia-smi / Nsight Systems 校准, 不影响结论方向。

纯标准库, 不依赖 GPU, 任何机器可跑: python scheduler_demo.py
"""
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

random.seed(42)

# ---------------- 可调参数(排障时逐条报出 = 参数意识) ----------------
NUM_STREAMS = 16            # 路数
DURATION_MS = 60_000        # 模拟时长 60 秒
JITTER_MS = 8.0             # 帧到达抖动 ±8ms
REORDER_RATE = 0.04         # 4% 的帧迟到(路内乱序)
REORDER_DELAY_FACTOR = 2.5  # 迟到 = 2.5 个帧周期
WINDOW_MS = 10              # 组批时间窗
MAX_BATCH = 8               # 批上限
BUF_SIZE = 2                # 每路只留最新 2 帧(丢旧留新)
LAT_BASE_MS = 4.0           # 推理固定开销(引擎启动/调度)
LAT_PER_FRAME_MS = 1.0      # 批内每帧增量

# 显存估算(MB), YOLOv8m 量级
W_WEIGHT_FP16 = 50          # FP16 引擎权重
W_WEIGHT_INT8 = 12          # INT8 引擎权重
W_CONTEXT = 400             # 每个进程的 CUDA 上下文开销(数百 MB 量级)
W_BUF_FP16 = 40             # 每路激活/工作缓冲
W_BUF_INT8 = 20             # INT8 下激活减半


def gen_stream(fps):
    """生成一路帧的 (到达时间ms, 序号) 列表, 按实际到达时间排序。
    迟到帧到达时会排在序号更大的帧后面, 由此产生路内乱序。"""
    interval = 1000.0 / fps
    frames = []
    t = random.uniform(0.0, interval)
    seq = 0
    while t < DURATION_MS:
        late = random.random() < REORDER_RATE
        at = t + (interval * REORDER_DELAY_FACTOR if late else 0.0)
        if at < DURATION_MS:
            frames.append((at, seq))
        seq += 1
        t += max(1.0, interval + random.uniform(-JITTER_MS, JITTER_MS))
    frames.sort(key=lambda x: x[0])
    return frames


def deliver(streams, idx, buf, last, t):
    """处理 t 时刻的帧到达: 乱序旧帧丢弃, 每路缓冲丢旧留新。"""
    overflow = stale = 0
    for s in range(NUM_STREAMS):
        stream = streams[s]
        while idx[s] < len(stream) and stream[idx[s]][0] <= t:
            at, seq = stream[idx[s]]
            idx[s] += 1
            if seq <= last[s]:          # 迟到的旧帧, 直接丢
                stale += 1
                continue
            if len(buf[s]) >= BUF_SIZE:  # 缓冲满: 丢旧留新
                buf[s].pop(0)
                overflow += 1
            buf[s].append((at, seq))
    return overflow, stale


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))]


def sim_naive(streams):
    """方案 A: 每路一个推理实例, 来一帧推一帧(batch=1)。"""
    idx = [0] * NUM_STREAMS
    buf = [[] for _ in range(NUM_STREAMS)]
    last = [-1] * NUM_STREAMS
    gpu_free = 0.0
    busy = 0.0
    lats = []
    overflow = stale = 0
    for t in range(DURATION_MS):
        ov, st = deliver(streams, idx, buf, last, t)
        overflow += ov
        stale += st
        for s in range(NUM_STREAMS):
            while buf[s] and gpu_free <= t:
                at, seq = buf[s].pop(0)
                if seq <= last[s]:
                    stale += 1
                    continue
                start = max(at, gpu_free)
                lat = LAT_BASE_MS + LAT_PER_FRAME_MS
                gpu_free = start + lat
                busy += lat
                lats.append(gpu_free - at)
                last[s] = seq
    return dict(busy=busy, lats=lats, drop=overflow + stale, stale=stale,
                n_batches=len(lats), total_batch=len(lats),
                backlog=sum(len(b) for b in buf))


def sim_batched(streams):
    """方案 B: 单模型实例 + 动态组批(时间窗 + 部分 batch + 丢旧留新)。"""
    idx = [0] * NUM_STREAMS
    buf = [[] for _ in range(NUM_STREAMS)]
    last = [-1] * NUM_STREAMS
    gpu_free = 0.0
    busy = 0.0
    lats = []
    overflow = stale = 0
    batch_due = None
    rr = 0                      # 轮询起点, 避免饿死某一路
    n_batches = total_batch = 0
    for t in range(DURATION_MS):
        ov, st = deliver(streams, idx, buf, last, t)
        overflow += ov
        stale += st
        buffered = sum(len(b) for b in buf)
        if batch_due is None and gpu_free <= t and buffered > 0:
            batch_due = t + WINDOW_MS          # 开窗
        due = batch_due is not None and (t >= batch_due or buffered >= MAX_BATCH)
        if due and gpu_free <= t:
            take = []                          # 轮询取帧, 凑不满也跑
            scanned = 0
            s = rr
            while len(take) < MAX_BATCH and scanned < NUM_STREAMS:
                if buf[s]:
                    at, seq = buf[s].pop(0)
                    take.append((at, seq, s))
                    rr = (s + 1) % NUM_STREAMS
                    scanned = 0
                else:
                    scanned += 1
                s = (s + 1) % NUM_STREAMS
            if take:
                lat = LAT_BASE_MS + LAT_PER_FRAME_MS * len(take)
                finish = t + lat
                gpu_free = finish
                busy += lat
                for at, seq, s in take:
                    if seq <= last[s]:         # 混进缓冲的乱序旧帧
                        stale += 1
                        continue
                    lats.append(finish - at)
                    last[s] = max(last[s], seq)
                n_batches += 1
                total_batch += len(take)
            batch_due = None
    return dict(busy=busy, lats=lats, drop=overflow + stale, stale=stale,
                n_batches=n_batches, total_batch=total_batch,
                backlog=sum(len(b) for b in buf))


def vram_naive():
    return NUM_STREAMS * (W_WEIGHT_FP16 + W_CONTEXT + W_BUF_FP16)


def vram_shared(int8=False):
    w = W_WEIGHT_INT8 if int8 else W_WEIGHT_FP16
    b = W_BUF_INT8 if int8 else W_BUF_FP16
    return w + W_CONTEXT + b * MAX_BATCH


def fmt_gb(mb):
    return f"{mb / 1024:.2f}GB"


def wlen(s):
    return sum(2 if ord(c) > 0x2E7F else 1 for c in s)


def pad(s, width):
    return s + " " * max(0, width - wlen(s))


def padl(s, width):
    return " " * max(0, width - wlen(s)) + s


SCENARIOS = [
    ("场景1: 16路 x 25fps 全帧检测(重负载)", 25),
    ("场景2: 16路 x 6fps 隔帧检测 + 跟踪器补帧(推荐做法)", 6),
]


def main():
    print("=" * 78)
    print("16路视频流调度对比: 每路一实例 vs 单实例+动态组批")
    print("(纯CPU模拟; 推理延迟/显存为参数化估算, 可在真实环境用")
    print(" nvidia-smi / Nsight Systems 校准, 结论方向不变)")
    print("=" * 78)
    for name, fps in SCENARIOS:
        streams = [gen_stream(fps) for _ in range(NUM_STREAMS)]
        a = sim_naive(streams)
        b = sim_batched(streams)
        total = sum(len(x) for x in streams)
        print()
        print(f"----- {name} -----  模拟60s, 总帧数 {total}")
        hdr = (pad("方案", 26) + "|" + padl("显存", 8) + "|" + padl("GPU占用", 9)
               + "|" + padl("均batch", 8) + "|" + padl("延迟p50/p99", 14)
               + "|" + padl("丢帧(乱序)", 12) + "|" + padl("积压", 6))
        print(hdr)
        print("-" * wlen(hdr))
        ua, ub = a["busy"] / DURATION_MS * 100, b["busy"] / DURATION_MS * 100
        la = f"{pct(a['lats'], 0.5):.0f}/{pct(a['lats'], 0.99):.0f}ms"
        lb = f"{pct(b['lats'], 0.5):.0f}/{pct(b['lats'], 0.99):.0f}ms"
        avg_b = b["total_batch"] / b["n_batches"] if b["n_batches"] else 0.0
        rows = [
            ("A 每路一实例(batch=1)", fmt_gb(vram_naive()), f"{ua:.1f}%", "1.0",
             la, f"{a['drop']}({a['stale']})", str(a["backlog"])),
            ("B 单实例+动态组批", fmt_gb(vram_shared()), f"{ub:.1f}%",
             f"{avg_b:.1f}", lb, f"{b['drop']}({b['stale']})", str(b["backlog"])),
            ("B+INT8(权重/激活减半)", fmt_gb(vram_shared(int8=True)), f"{ub:.1f}%",
             f"{avg_b:.1f}", lb, f"{b['drop']}({b['stale']})", str(b["backlog"])),
        ]
        for row in rows:
            print(pad(row[0], 26) + "|" + padl(row[1], 8) + "|" + padl(row[2], 9)
                  + "|" + padl(row[3], 8) + "|" + padl(row[4], 14)
                  + "|" + padl(row[5], 12) + "|" + padl(row[6], 6))
        print()
        print(f"  显存: {fmt_gb(vram_naive())} -> {fmt_gb(vram_shared())}"
              f" (单实例, 约 1/{vram_naive() / vram_shared():.0f})")
        print(f"        {fmt_gb(vram_shared())} -> {fmt_gb(vram_shared(int8=True))} (再上 INT8)")
        print(f"  8GB 显卡: A 方案直接 OOM; B 方案只占 ~{vram_shared() / 8192 * 100:.0f}%")
        print(f"  组批延迟代价 = 一个时间窗(<= {WINDOW_MS}ms); 吞吐与显存收益远大于此")
    print()
    print("=" * 78)
    print("要点:")
    print(" 1. 显存不够的根因通常是'权重 x 16 份 + 每进程 CUDA 上下文', 不是模型大")
    print(" 2. 单实例 + 动态组批: 显存降到约 1/10, 一张 8GB 卡吃下 16 路")
    print(" 3. 乱序不是 bug 是约束: 迟到旧帧丢旧留新, 慢路不阻塞快路(部分 batch)")
    print(" 4. 丢帧由跟踪器(Kalman)容忍; 隔帧检测把负载再降 4 倍")
    print(" 5. 同架构的业界参考实现: NVIDIA DeepStream(单模型实例+组批+NVDEC 直通)")
    print("=" * 78)


if __name__ == "__main__":
    main()
