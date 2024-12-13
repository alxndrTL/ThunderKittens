import sys
import time

import torch
import thunderkittens as tk

B=16
H=12
N=int(sys.argv[1]) if len(sys.argv) > 1 else 1024
D=64
iters = 10

torch.random.manual_seed(42)
q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5)
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5)
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D)

# warmup
o = tk.lin_attn(q, k, v)


st = time.time()
for _ in range(iters):
    o = tk.lin_attn(q, k, v)
et = time.time()

dt = (et - st)/iters

print(f"{dt * 1e6:.2f} µs")
