# -*- coding: utf-8 -*-
"""
16 路视频流调度器模拟 —— **重构版**（与 before_scheduler_demo.py 行为完全一致）
==============================================================================

重构手法（能力点「代码重构」）：
  1. 散落的模块级常量 -> 一个 frozen `Config`（唯一真源，可传参、可测试）
  2. `sim_naive` / `sim_batched` 里重复的“取帧/入缓冲/丢旧留新” -> 抽成 `SimState.drain()`
  3. 两个方案共用的统计装配 -> `SimState.metrics()`
  4. `main()` 里“模拟 + 排版 + 结论话术”混在一起 -> 拆成 `render_scenario()` / `render_summary()`
  5. 方案 A/B 各自的状态 -> 统一 `SimState`，A/B 只保留**策略差异**

验收：`verify.sh` 断言重构前后 stdout **逐字节一致**。
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# --------------------------------------------------------------------------
# 配置：原先散在模块顶部的 14 个常量，收拢为唯一真源
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    streams: int = 16
    duration_ms: int = 60_000
    jitter_ms: float = 8.0
    reorder_rate: float = 0.04
    reorder_delay_factor: float = 2.5
    window_ms: int = 10
    max_batch: int = 8
    buf_size: int = 2
    lat_base_ms: float = 4.0
    lat_per_frame_ms: float = 1.0
    w_weight_fp16: int = 50
    w_weight_int8: int = 12
    w_context: int = 400
    w_buf_fp16: int = 40
    w_buf_int8: int = 20


def gen_stream(cfg: Config, fps: float):
    """生成一路帧的 (到达时间ms, 序号)，按到达时间排序。逻辑与原版一致。"""
    interval = 1000.0 / fps
    frames = []
    t = random.uniform(0.0, interval)
    seq = 0
    while t < cfg.duration_ms:
        late = random.random() < cfg.reorder_rate
        at = t + (interval * cfg.reorder_delay_factor if late else 0.0)
        if at < cfg.duration_ms:
            frames.append((at, seq))
        seq += 1
        t += max(1.0, interval + random.uniform(-cfg.jitter_ms, cfg.jitter_ms))
    frames.sort(key=lambda x: x[0])
    return frames


@dataclass
class SimState:
    """两个方案共享的调度状态；策略差异留在 run_* 里。"""
    cfg: Config
    streams: list
    idx: list = field(default_factory=list)
    buf: list = field(default_factory=list)
    last: list = field(default_factory=list)
    gpu_free: float = 0.0
    busy: float = 0.0
    lats: list = field(default_factory=list)
    overflow: int = 0
    stale: int = 0
    n_batches: int = 0
    total_batch: int = 0
    batch_due: int | None = None
    rr: int = 0

    def __post_init__(self):
        n = self.cfg.streams
        self.idx = [0] * n
        self.buf = [[] for _ in range(n)]
        self.last = [-1] * n

    def drain(self, t):
        """t 时刻的到达处理：迟到的旧帧丢弃，每路缓冲丢旧留新。（原 deliver()）"""
        cfg = self.cfg
        for s in range(cfg.streams):
            stream = self.streams[s]
            while self.idx[s] < len(stream) and stream[self.idx[s]][0] <= t:
                at, seq = stream[self.idx[s]]
                self.idx[s] += 1
                if seq <= self.last[s]:
                    self.stale += 1
                    continue
                if len(self.buf[s]) >= cfg.buf_size:
                    self.buf[s].pop(0)
                    self.overflow += 1
                self.buf[s].append((at, seq))

    def buffered(self):
        return sum(len(b) for b in self.buf)

    def metrics(self):
        return dict(busy=self.busy, lats=self.lats, drop=self.overflow + self.stale,
                    stale=self.stale, n_batches=self.n_batches,
                    total_batch=self.total_batch, backlog=self.buffered())


def run_naive(state: SimState):
    """方案 A：每路一个推理实例，来一帧推一帧(batch=1)。"""
    cfg = state.cfg
    for t in range(cfg.duration_ms):
        state.drain(t)
        for s in range(cfg.streams):
            while state.buf[s] and state.gpu_free <= t:
                at, seq = state.buf[s].pop(0)
                if seq <= state.last[s]:
                    state.stale += 1
                    continue
                start = max(at, state.gpu_free)
                lat = cfg.lat_base_ms + cfg.lat_per_frame_ms
                state.gpu_free = start + lat
                state.busy += lat
                state.lats.append(state.gpu_free - at)
                state.last[s] = seq
    return state.metrics()


def run_batched(state: SimState):
    """方案 B：单模型实例 + 动态组批(时间窗 + 部分 batch + 丢旧留新)。"""
    cfg = state.cfg
    for t in range(cfg.duration_ms):
        state.drain(t)
        buffered = state.buffered()
        if state.batch_due is None and state.gpu_free <= t and buffered > 0:
            state.batch_due = t + cfg.window_ms
        due = state.batch_due is not None and (t >= state.batch_due or buffered >= cfg.max_batch)
        if due and state.gpu_free <= t:
            take = []
            scanned = 0
            s = state.rr
            while len(take) < cfg.max_batch and scanned < cfg.streams:
                if state.buf[s]:
                    at, seq = state.buf[s].pop(0)
                    take.append((at, seq, s))
                    state.rr = (s + 1) % cfg.streams
                    scanned = 0
                else:
                    scanned += 1
                s = (s + 1) % cfg.streams
            if take:
                lat = cfg.lat_base_ms + cfg.lat_per_frame_ms * len(take)
                finish = t + lat
                state.gpu_free = finish
                state.busy += lat
                for at, seq, s in take:
                    if seq <= state.last[s]:
                        state.stale += 1
                        continue
                    state.lats.append(finish - at)
                    state.last[s] = max(state.last[s], seq)
                state.n_batches += 1
                state.total_batch += len(take)
            state.batch_due = None
    return state.metrics()


# --------------------------------------------------------------------------
# 显存模型
# --------------------------------------------------------------------------
def vram_naive(cfg: Config):
    return cfg.streams * (cfg.w_weight_fp16 + cfg.w_context + cfg.w_buf_fp16)


def vram_shared(cfg: Config, int8: bool = False):
    w = cfg.w_weight_int8 if int8 else cfg.w_weight_fp16
    b = cfg.w_buf_int8 if int8 else cfg.w_buf_fp16
    return w + cfg.w_context + b * cfg.max_batch


# --------------------------------------------------------------------------
# 排版（中英文混排宽度）
# --------------------------------------------------------------------------
def fmt_gb(mb):
    return f"{mb / 1024:.2f}GB"


def wlen(s):
    return sum(2 if ord(c) > 0x2E7F else 1 for c in s)


def pad(s, width):
    return s + " " * max(0, width - wlen(s))


def padl(s, width):
    return " " * max(0, width - wlen(s)) + s


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))]


SCENARIOS = [
    ("场景1: 16路 x 25fps 全帧检测(重负载)", 25),
    ("场景2: 16路 x 6fps 隔帧检测 + 跟踪器补帧(推荐做法)", 6),
]


def rows_for(cfg, a, b, ua, ub):
    la = f"{pct(a['lats'], 0.5):.0f}/{pct(a['lats'], 0.99):.0f}ms"
    lb = f"{pct(b['lats'], 0.5):.0f}/{pct(b['lats'], 0.99):.0f}ms"
    avg_b = b["total_batch"] / b["n_batches"] if b["n_batches"] else 0.0
    return [
        ("A 每路一实例(batch=1)", fmt_gb(vram_naive(cfg)), f"{ua:.1f}%", "1.0",
         la, f"{a['drop']}({a['stale']})", str(a["backlog"])),
        ("B 单实例+动态组批", fmt_gb(vram_shared(cfg)), f"{ub:.1f}%",
         f"{avg_b:.1f}", lb, f"{b['drop']}({b['stale']})", str(b["backlog"])),
        ("B+INT8(权重/激活减半)", fmt_gb(vram_shared(cfg, int8=True)), f"{ub:.1f}%",
         f"{avg_b:.1f}", lb, f"{b['drop']}({b['stale']})", str(b["backlog"])),
    ]


def render_scenario(cfg, name, fps):
    streams = [gen_stream(cfg, fps) for _ in range(cfg.streams)]
    a = run_naive(SimState(cfg, streams))
    b = run_batched(SimState(cfg, streams))
    total = sum(len(x) for x in streams)
    print()
    print(f"----- {name} -----  模拟60s, 总帧数 {total}")
    hdr = (pad("方案", 26) + "|" + padl("显存", 8) + "|" + padl("GPU占用", 9)
           + "|" + padl("均batch", 8) + "|" + padl("延迟p50/p99", 14)
           + "|" + padl("丢帧(乱序)", 12) + "|" + padl("积压", 6))
    print(hdr)
    print("-" * wlen(hdr))
    ua, ub = a["busy"] / cfg.duration_ms * 100, b["busy"] / cfg.duration_ms * 100
    for row in rows_for(cfg, a, b, ua, ub):
        print(pad(row[0], 26) + "|" + padl(row[1], 8) + "|" + padl(row[2], 9)
              + "|" + padl(row[3], 8) + "|" + padl(row[4], 14)
              + "|" + padl(row[5], 12) + "|" + padl(row[6], 6))
    print()
    print(f"  显存: {fmt_gb(vram_naive(cfg))} -> {fmt_gb(vram_shared(cfg))}"
          f" (单实例, 约 1/{vram_naive(cfg) / vram_shared(cfg):.0f})")
    print(f"        {fmt_gb(vram_shared(cfg))} -> {fmt_gb(vram_shared(cfg, int8=True))} (再上 INT8)")
    print(f"  8GB 显卡: A 方案直接 OOM; B 方案只占 ~{vram_shared(cfg) / 8192 * 100:.0f}%")
    print(f"  组批延迟代价 = 一个时间窗(<= {cfg.window_ms}ms); 吞吐与显存收益远大于此")


def render_header():
    print("=" * 78)
    print("16路视频流调度对比: 每路一实例 vs 单实例+动态组批")
    print("(纯CPU模拟; 推理延迟/显存为参数化估算, 可在真实环境用")
    print(" nvidia-smi / Nsight Systems 校准, 结论方向不变)")
    print("=" * 78)


def render_summary():
    print()
    print("=" * 78)
    print("要点:")
    print(" 1. 显存不够的根因通常是'权重 x 16 份 + 每进程 CUDA 上下文', 不是模型大")
    print(" 2. 单实例 + 动态组批: 显存降到约 1/10, 一张 8GB 卡吃下 16 路")
    print(" 3. 乱序不是 bug 是约束: 迟到旧帧丢旧留新, 慢路不阻塞快路(部分 batch)")
    print(" 4. 丢帧由跟踪器(Kalman)容忍; 隔帧检测把负载再降 4 倍")
    print(" 5. 同架构的业界参考实现: NVIDIA DeepStream(单模型实例+组批+NVDEC 直通)")
    print("=" * 78)


def main(cfg: Config = Config()):
    random.seed(42)
    render_header()
    for name, fps in SCENARIOS:
        render_scenario(cfg, name, fps)
    render_summary()


if __name__ == "__main__":
    main()
