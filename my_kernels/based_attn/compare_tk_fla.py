# -*- coding: utf-8 -*-

from typing import Optional

import torch
from einops import rearrange


import torch
from tqdm import trange
import numpy as np
import sys
from einops import rearrange
import math

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
N = 1024
D = 16
DV = 64

TESTNAME = sys.argv[1]

if TESTNAME in ['ones_all', 'ones_t0', 'ones_t1', 'ones_t0t1', 'ones_t2']:
    q = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
    k = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
    v = (torch.ones((B, H, N, DV), dtype=torch.bfloat16, device='cuda')/DV).to(torch.float32)
elif TESTNAME in ['randn', 'randn_all', 'randn_t0', 'randn_t1', 'randn_t0t1', 'randn_t2']:
    torch.random.manual_seed(42)
    q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    v = (torch.randn((B, H, N, DV), dtype=torch.bfloat16, device='cuda')/DV).to(torch.float32)
else:
    print('Invalid test name')
    sys.exit(0)

def pytorch_test_kittens(Q, K, V, add_scale = True):
    print("Adding scale:", add_scale)
    B, H, L, D = Q.shape

    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X

    O   = torch.einsum("bhnd,bhmd->bhnm", Q.to(torch.float32), K.to(torch.float32))**2
    O2  = make_causal(O)
    T2  = torch.einsum(
        "bhnm,bhmd->bhnd", 
        O2.to(torch.float32), 
        V.to(torch.float32)
    ).to(torch.float32)
    T1a = make_causal(
        torch.einsum("bhnd,bhmd->bhnm", 
        Q.to(torch.float32), K.to(torch.float32))
    )
    T1 = torch.einsum(
        "bhnm,bhme->bhne", 
        T1a.to(torch.float32), V.to(torch.float32)
    ).to(torch.float32)
    T0  = V.to(torch.float32).cumsum(dim=2)

    rd  = math.sqrt(D) if add_scale else 1    
    rrd = math.sqrt(rd) if add_scale else 1   
    r2  = math.sqrt(2) if add_scale else 1

    o = 0
    o += T0.to(torch.float32)
    o += T1.to(torch.float32) / (rrd * rrd)
    o += T2.to(torch.float32) / (rd * r2 * rd * r2)
    return o#.to(torch.bfloat16)

def pytorch_test_fla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
):
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q = q * scale
    attn = q @ k.transpose(-2, -1)
    attn = 1 + attn + 1/2 * (attn ** 2)
    attn.masked_fill_(~torch.tril(torch.ones(
        q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device)), 0)
    o = attn @ v
    return o

o_kittens = pytorch_test_kittens(q, k, v)
o_fla = pytorch_test_fla(q, k, v)

print(torch.allclose(o_kittens, o_fla, atol=0.001))



"""
with open(f'fla_{TESTNAME}_{B}x{H}x{N}x{D}x{DV}.txt', 'w') as f:
    qf = q.to(torch.float32).flatten().cpu().numpy().tolist()
    kf = k.to(torch.float32).flatten().cpu().numpy().tolist()
    vf = v.to(torch.float32).flatten().cpu().numpy().tolist()
    of = o.to(torch.float32).flatten().cpu().numpy().tolist()

    #for i in trange(B*H*N*D):
    #    f.write(repr(qf[i]))
    #    f.write(' ')
    #for i in trange(B*H*N*D):
    #    f.write(repr(kf[i]))
    #    f.write(' ')
    #for i in trange(B*H*N*DV):
    #    f.write(repr(vf[i]))
    #    f.write(' ')
    for i in trange(B*H*N*DV):
        f.write(repr(of[i]))
        f.write(' ')
"""