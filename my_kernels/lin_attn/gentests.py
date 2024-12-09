import torch
from tqdm import trange
import numpy as np
import sys
from einops import rearrange
import math

"""
python gentests.py randn 1024 64
"""

from typing import Optional, Tuple

def naive_chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    chunk_size = 64
    q = rearrange(q, 'b h (n c) d -> b h n c d', c=chunk_size) * q.shape[-1] ** -0.5
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size)
    kv = k.transpose(-1, -2) @ v
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    inter = q @ kv
    intra = (q @ k.transpose(-1, -2)) @ v
    o = inter + intra
    return rearrange(o, 'b h n c d -> b h (n c) d')

# Parameters
B = 1
H = 1
N = 1024 if len(sys.argv) <= 2 else int(sys.argv[2])
D = 128 if len(sys.argv) <= 3 else int(sys.argv[3])

TESTNAME = sys.argv[1]

if TESTNAME == 'ones':
    q = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'qk_test':
    q = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    q = q.to(dtype=torch.bfloat16, device='cuda')
    
    k = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    k = k.to(dtype=torch.bfloat16, device='cuda')
    
    v = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    v = v.to(dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'v_orientation':
    q = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v = (torch.arange(D, dtype=torch.bfloat16, device='cuda')/D).reshape((1,1,1,-1)).repeat(B, H, N, 1)
else:
    print('Invalid test name')
    sys.exit(0)

# Compute linear attention
o = naive_chunk_linear_attn(q, k, v)

fn = f'{TESTNAME}_linear_{N}_{D}.txt'
with open(fn, 'w') as f:
    # Flatten and write tensors
    qf = rearrange(q, "b h n d -> b n h d").to(torch.float32).flatten().detach().cpu().numpy()
    kf = rearrange(k, "b h n d -> b n h d").to(torch.float32).flatten().detach().cpu().numpy()
    vf = rearrange(v, "b h n d -> b n h d").to(torch.float32).flatten().detach().cpu().numpy()
    of = rearrange(o, "b h n d -> b n h d").to(torch.float32).flatten().detach().cpu().numpy()
    
    for i in trange(B*H*N*D):
        f.write(repr(qf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(kf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(vf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(of[i]))
        f.write(' ')
