import math

import torch
import torch.nn.functional as F
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_seq_len=10000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb


def apply_rotary_pos_emb(t, freqs):
    if freqs.dim() == 2:
        freqs = freqs.unsqueeze(0).unsqueeze(2)
    elif freqs.dim() == 3:
        freqs = freqs.unsqueeze(2)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, norm=nn.RMSNorm, use_qk_norm=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.qkv_attn = nn.Linear(embed_dim, embed_dim * 3)
        self.project = nn.Linear(embed_dim, embed_dim)
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = norm(self.head_dim)
            self.k_norm = norm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rotary_freqs=None,
        attn_mask=None,
        rotary_start_index: int = 0,
    ):
        b, n, c = x.size()
        qkv = self.qkv_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        q = q.view(b, n, self.num_heads, self.head_dim)
        k = k.view(b, n, self.num_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rotary_freqs is not None:
            if rotary_start_index < 0 or rotary_start_index > n:
                raise ValueError(
                    f"rotary_start_index={rotary_start_index} is invalid for sequence length={n}."
                )
            if rotary_start_index == 0:
                q = apply_rotary_pos_emb(q, rotary_freqs)
                k = apply_rotary_pos_emb(k, rotary_freqs)
            elif rotary_start_index < n:
                q_vis = apply_rotary_pos_emb(q[:, rotary_start_index:, :, :], rotary_freqs)
                k_vis = apply_rotary_pos_emb(k[:, rotary_start_index:, :, :], rotary_freqs)
                q = torch.cat([q[:, :rotary_start_index, :, :], q_vis], dim=1)
                k = torch.cat([k[:, :rotary_start_index, :, :], k_vis], dim=1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
        )
        y = y.transpose(1, 2).contiguous().view(b, n, c)
        y = self.project(y)
        return y, None


class AdaptivePreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fn = fn

    def forward(self, x, shift, scale, gate, **kwargs):
        out = self.fn(modulate(self.norm(x), shift, scale), **kwargs)
        if isinstance(self.fn, Attention):
            f, _ = out
        else:
            f = out
        return gate * f


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
