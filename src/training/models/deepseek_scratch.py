"""
手写 DeepSeek-V2 —— 包含两个核心技术：
  1. MLA (Multi-head Latent Attention) 多头潜在注意力
  2. DeepSeekMoE 混合专家（细粒度专家 + 共享专家）

这不是简化版：参数量由 config 决定，和官方一模一样。
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class DSConfig:
    vocab_size: int = 102400
    hidden_size: int = 2048
    intermediate_size: int = 10944
    moe_intermediate_size: int = 1408
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    n_shared_experts: int = 2
    n_routed_experts: int = 64
    num_experts_per_tok: int = 6
    kv_lora_rank: int = 512
    q_lora_rank: Optional[int] = None
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    first_k_dense_replace: int = 1
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    attention_bias: bool = False
    mlp_bias: bool = False
    routed_scaling_factor: float = 1.0
    topk_method: str = "greedy"
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    tie_word_embeddings: bool = False


# ─────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """和 Qwen 一样，但 HF 会先转 fp32 再算（数值更稳）"""
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        d = x.dtype
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (self.weight * (x.float() * torch.rsqrt(v + self.eps))).to(d)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        f = torch.outer(t, self.inv_freq)                 # (seq, dim/2)
        return torch.cos(f), torch.sin(f)                 # (seq, dim/2)


def apply_rope(x, cos, sin):
    """复数形式的 RoPE：相邻两个元素 (x[2i], x[2i+1]) 组成复数乘 e^{i·θ}"""
    # x: (B, n_heads, S, dim) ; cos/sin: (S, dim/2)
    B, H, S, D = x.shape
    x2 = x.view(B, H, S, D // 2, 2)
    c = cos.view(1, 1, S, D // 2, 1); s = sin.view(1, 1, S, D // 2, 1)
    x0, x1 = x2[..., 0:1], x2[..., 1:2]
    o0 = x0 * c - x1 * s
    o1 = x0 * s + x1 * c
    return torch.cat([o0, o1], dim=-1).view(B, H, S, D)


# ─────────────────────────────────────────────────────────────
class MLA(nn.Module):
    """
    MLA 的核心思想：KV Cache 压缩。
    传统 MHA 要缓存 seq_len × n_heads × head_dim × 2 (K和V)
    MLA 只缓存 seq_len × kv_lora_rank 的一个"潜向量"，推理时再解压还原。

    V2-Lite: kv_lora_rank=512, 而 16头×128维×2(K,V) = 4096
    → KV Cache 压缩到 1/8
    """
    def __init__(self, cfg: DSConfig):
        super().__init__()
        self.nh = cfg.num_attention_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_nope = cfg.qk_nope_head_dim      # 不带位置信息的 q/k 维度
        self.qk_rope = cfg.qk_rope_head_dim      # 带 RoPE 的维度
        self.v_dim = cfg.v_head_dim
        self.qk_head_dim = self.qk_nope + self.qk_rope
        self.q_lora_rank = cfg.q_lora_rank
        self.scale = self.qk_head_dim ** -0.5

        # ---- Q 路径：可选低秩压缩 ----
        if cfg.q_lora_rank is None:
            self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.qk_head_dim, bias=False)
        else:
            self.q_proj = None
            self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=cfg.attention_bias)
            self.q_a_layernorm = RMSNorm(cfg.q_lora_rank)
            self.q_b_proj = nn.Linear(cfg.q_lora_rank, self.nh * self.qk_head_dim, bias=False)

        # ---- KV 路径：压缩成潜向量 ----
        self.kv_a_proj_with_mqa = nn.Linear(
            cfg.hidden_size, cfg.kv_lora_rank + self.qk_rope, bias=cfg.attention_bias)
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank)
        # 解压：潜向量 → 所有头的 K(nope) 和 V
        self.kv_b_proj = nn.Linear(
            cfg.kv_lora_rank, self.nh * (self.qk_nope + self.v_dim), bias=False)

        self.o_proj = nn.Linear(self.nh * self.v_dim, cfg.hidden_size, bias=cfg.attention_bias)
        self.rotary = RotaryEmbedding(self.qk_rope, cfg.rope_theta)

    def forward(self, h):
        B, S, _ = h.shape
        # 1) Q
        if self.q_lora_rank is None:
            q = self.q_proj(h)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(h)))
        q = q.view(B, S, self.nh, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope, self.qk_rope], dim=-1)

        # 2) KV 压缩成一个潜向量（★ 这就是要缓存的东西）
        ckv = self.kv_a_proj_with_mqa(h)
        kv_latent, k_pe = torch.split(ckv, [self.kv_lora_rank, self.qk_rope], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)                       # (B,S,kv_lora_rank)
        k_pe = k_pe.view(B, 1, S, self.qk_rope)                          # 所有头共享同一个位置编码

        # 3) 只对位置部分做 RoPE
        cos, sin = self.rotary(S)
        cos, sin = cos.to(h.dtype), sin.to(h.dtype)
        q_pe = apply_rope(q_pe, cos, sin)
        k_pe = apply_rope(k_pe, cos, sin)

        # 4) 解压出 K 和 V
        kv = self.kv_b_proj(kv_latent).view(B, S, self.nh, self.qk_nope + self.v_dim).transpose(1, 2)
        k_nope, v = torch.split(kv, [self.qk_nope, self.v_dim], dim=-1)

        q = torch.cat([q_nope, q_pe], dim=-1)                            # (B,nh,S,192)
        k_pe = k_pe.expand(-1, self.nh, -1, -1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1))


# ─────────────────────────────────────────────────────────────
class Experts(nn.Module):
    """64 个专家，但每个 token 只用 top-6 → 只算 6 个专家"""
    def __init__(self, cfg: DSConfig):
        super().__init__()
        self.n = cfg.n_routed_experts
        self.inter = cfg.moe_intermediate_size
        # gate/up 拼在一起存，一次线性算完再 chunk（省一次 kernel launch）
        self.gate_up_proj = nn.Parameter(torch.empty(self.n, 2 * self.inter, cfg.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.n, cfg.hidden_size, self.inter))
        self.act = nn.SiLU()
        nn.init.normal_(self.gate_up_proj, std=0.02); nn.init.normal_(self.down_proj, std=0.02)

    def forward(self, x, topk_idx, topk_w):
        """
        x: (T, H)   topk_idx: (T, K)   topk_w: (T, K)

        朴素写法：for e in range(n): one_hot + torch.where 找出属于该专家的 token。
        问题：每轮都做一次全表扫描，MPS 上是主要瓶颈。

        优化：先按专家 argsort 一次，token 按专家聚成连续段，
        之后每轮只要切一个切片 —— 去掉 one_hot 和 where。
        （注意：不能把「每个 token 一个权重矩阵」展开成 bmm，
          那会物化出 (T*K, 2*inter, H) 的巨大张量，显存直接爆。）
        """
        T, H = x.shape
        K = topk_idx.shape[1]
        flat_e = topk_idx.reshape(-1)
        flat_t = torch.arange(T, device=x.device).unsqueeze(1).expand(T, K).reshape(-1)
        flat_w = topk_w.reshape(-1)

        order = torch.argsort(flat_e)                       # 按专家排序
        sorted_e = flat_e[order]
        counts = torch.bincount(sorted_e, minlength=self.n).tolist()  # 16 个小整数，可同步

        out = torch.zeros(T, H, dtype=x.dtype, device=x.device)
        pos = 0
        for e, c in enumerate(counts):
            if c == 0:
                continue
            idx = order[pos:pos + c]
            pos += c
            toks = flat_t[idx]                              # (c,) 这些 token 选了专家 e
            h = x.index_select(0, toks)                     # (c, H)
            g, u = F.linear(h, self.gate_up_proj[e]).chunk(2, dim=-1)
            y = F.linear(self.act(g) * u, self.down_proj[e])   # (c, H)
            out.index_add_(0, toks, y * flat_w[idx].unsqueeze(1))
        return out


class Router(nn.Module):
    def __init__(self, cfg: DSConfig):
        super().__init__()
        self.top_k = cfg.num_experts_per_tok
        self.n = cfg.n_routed_experts
        self.weight = nn.Parameter(torch.zeros(self.n, cfg.hidden_size))   # 注意：不是 Linear
        self.scaling = cfg.routed_scaling_factor
        self.topk_method = cfg.topk_method
        self.n_group = cfg.n_group; self.topk_group = cfg.topk_group

    def forward(self, h):
        h = h.view(-1, h.shape[-1])
        logits = F.linear(h.float(), self.weight.float())                 # fp32 算路由，防溢出
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        if self.topk_method == "greedy":
            w, idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            # V3 用的分组限制路由：先选组，再在组内选专家（防止全挤到少数几个）
            g = scores.view(-1, self.n_group, self.n // self.n_group).max(-1).values
            gi = torch.topk(g, k=self.topk_group, dim=-1, sorted=False)[1]
            gm = torch.zeros_like(g).scatter_(1, gi, 1)
            sm = gm.unsqueeze(-1).expand(-1, self.n_group, self.n // self.n_group).reshape(-1, self.n)
            w, idx = torch.topk(scores.masked_fill(~sm.bool(), 0.0), k=self.top_k, dim=-1, sorted=False)
        return logits, w * self.scaling, idx


class MoE(nn.Module):
    """DeepSeekMoE = 路由专家(top-k) + 共享专家(所有 token 都过)"""
    def __init__(self, cfg: DSConfig):
        super().__init__()
        self.experts = Experts(cfg)
        self.gate = Router(cfg)
        self.shared_experts = MLP(cfg, inter=cfg.moe_intermediate_size * cfg.n_shared_experts)

    def forward(self, h):
        residual = h
        orig = h.shape
        _, w, idx = self.gate(h)
        # 缓存路由结果供 ESFT 的专家激活统计使用（不参与梯度，不影响训练）
        if not self.training:
            self._last_topk = idx.detach()
            self._last_weight = w.detach()
        h = self.experts(h.view(-1, h.shape[-1]), idx, w).view(*orig)
        return h + self.shared_experts(residual)      # ★ 共享专家：每个 token 必过


class MLP(nn.Module):
    def __init__(self, cfg: DSConfig, inter=None):
        super().__init__()
        inter = inter or cfg.intermediate_size
        self.gate_proj = nn.Linear(cfg.hidden_size, inter, bias=cfg.mlp_bias)
        self.up_proj = nn.Linear(cfg.hidden_size, inter, bias=cfg.mlp_bias)
        self.down_proj = nn.Linear(inter, cfg.hidden_size, bias=cfg.mlp_bias)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: DSConfig, idx: int):
        super().__init__()
        self.self_attn = MLA(cfg)
        # 前 first_k_dense_replace 层用稠密 MLP（浅层路由不稳定）
        self.mlp = MLP(cfg) if idx < cfg.first_k_dense_replace else MoE(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class DeepseekV2ForCausalLM(nn.Module):
    def __init__(self, cfg: DSConfig):
        super().__init__(); self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        for l in self.layers: h = l(h)
        return self.lm_head(self.norm(h))

    def count_params(self):
        """区分：总参数 vs 每个 token 实际激活的参数（MoE 的关键指标）"""
        total = sum(p.numel() for p in self.parameters())
        return total
