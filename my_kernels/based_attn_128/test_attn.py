import sys
import time

import torch
import torch.nn.functional as F

B=2
H=6
N=int(sys.argv[1])
D=128
iters = 10

torch.random.manual_seed(42)
q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)

# warmup
o = F.scaled_dot_product_attention(q, k, v, is_causal=True)

st = time.time()
for _ in range(iters):
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
et = time.time()

dt = (et - st)/iters

print(f"{dt * 1e6:.2f} µs")


