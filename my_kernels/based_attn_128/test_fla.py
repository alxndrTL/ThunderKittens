import sys
import time

import torch
from fla.ops.linear_attn.fused_chunk import fused_chunk_linear_attn

B=16
H=6
N=int(sys.argv[1])
D=128
iters = 10

torch.random.manual_seed(42)
q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)

# warmup
o = fused_chunk_linear_attn(q, k, v, scale=1)

st = time.time()
for _ in range(iters):
    o = fused_chunk_linear_attn(q, k, v, scale=1)
et = time.time()

dt = (et - st)/iters

print(f"{dt * 1e6:.2f} µs")