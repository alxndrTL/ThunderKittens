import sys
import time

import torch
import thunderkittens as tk
import matplotlib.pyplot as plt  # added for plotting

from naive import python_naive, python_chunk

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
o = tk.lin_attn(q, k, v) # (B, H, N, D)

o_python_naive = python_naive(q.float(), k.float(), v.float())
o_python_chunk = python_chunk(q.float(), k.float(), v.float())

print(q.mean().item())
print(k.mean().item())
print(v.mean().item())
print(o.mean().item())

#print(torch.allclose(o_python_naive, o, atol=1e-2))

# Calculate mean error across B, H, and D dimensions for each index in N
mean_error = torch.abs(o_python_naive - o_python_chunk).mean(dim=(0,1,3)).float().cpu()

plt.figure()
plt.plot(mean_error.numpy())
plt.xlabel('N index')
plt.ylabel('Mean Error')
plt.title('Mean Error over N dimension')
plt.savefig('mean_error_pythonnaive_vs_pythonchunk.png')

st = time.time()
for _ in range(iters):
    o = tk.lin_attn(q, k, v)
et = time.time()

dt = (et - st)/iters

print(f"{dt * 1e6:.2f} µs")

