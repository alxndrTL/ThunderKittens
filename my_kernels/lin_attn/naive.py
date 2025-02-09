import torch
from einops import rearrange

def python_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    attn = q @ k.transpose(-2, -1)
    attn.masked_fill_(~torch.tril(torch.ones(q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device)), 0)
    o = attn @ v
    return o

def python_chunk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    chunk_size = 64
    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    kv = k.transpose(-1, -2) @ v
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    inter = q @ kv
    intra = ((
        q @ k.transpose(-1, -2)).masked_fill_(
        torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1),
        0
    )) @ v
    o = inter + intra
    return rearrange(o, 'b h n c d -> b h (n c) d')
