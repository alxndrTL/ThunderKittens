import sys
import time

import torch
import matplotlib.pyplot as plt  # added for plotting
from fla.ops.linear_attn.fused_chunk import fused_chunk_linear_attn

from naive import python_naive, python_chunk

B=16
H=12
N=int(sys.argv[1]) if len(sys.argv) > 1 else 2048
D=64
iters = 10

torch.random.manual_seed(42)
q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5)
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5)
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D)

# warmup
o, _ = fused_chunk_linear_attn(q, k, v, scale=1, normalize=False) # (B, H, N, D)

o_python_naive = python_naive(q, k, v)
o_python_chunk = python_chunk(q, k, v)

print(q.mean().item())
print(k.mean().item())
print(v.mean().item())
print(o_python_chunk.mean().item())
print(o.mean().item())

print(torch.allclose(o_python_naive, o, atol=1e-2))

# Calculate mean error across B, H, and D dimensions for each index in N
mean_error = torch.abs(o_python_chunk - o).mean(dim=(0,1,3)).float().cpu()

plt.figure()
plt.plot(mean_error.numpy())
plt.xlabel('N index')
plt.ylabel('Mean Error')
plt.title('Mean Error over N dimension')
#plt.savefig('mean_error_pythonnaive_vs_fla.png')

st = time.time()
for _ in range(iters):
    o, _ = fused_chunk_linear_attn(q, k, v, scale=1, normalize=False)
et = time.time()

dt = (et - st)/iters

print(f"{dt * 1e6:.2f} µs")


