import sys
import time

import torch
import thunderkittens as tk
import matplotlib.pyplot as plt  # added for plotting

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
q.requires_grad = True
k.requires_grad = True
v.requires_grad = True

# warmup
o = tk.lin_attn(q, k, v) # (B, H, N, D)

o_python_naive = python_naive(q, k, v)
o_python_chunk = python_chunk(q, k, v)

print(torch.allclose(o_python_naive, o, atol=1e-2))
mean_error = torch.abs(o_python_chunk - o).mean(dim=(0,1,3)).float().cpu()

J = o_python_naive.sum()
J.backward()

dq, dk, dv = tk.lin_attn_bwd(q, k, v, torch.ones_like(o))

print(torch.allclose(q.grad, dq, atol=1e-1))
print(torch.allclose(k.grad, dk, atol=1e-1))
print(torch.allclose(v.grad, dv, atol=1e-1))

#plt.figure()
#plt.plot(mean_error.numpy())
#plt.xlabel('N index')
#plt.ylabel('Mean Error')
#plt.title('Mean Error over N dimension')
#plt.savefig('mean_error_pythonnaive_vs_tk.png')

st = time.time()
for _ in range(iters):
    dq, dk, dv = tk.lin_attn_bwd(q, k, v, torch.ones_like(o))
et = time.time()

dt = (et - st)/iters

print(f"{dt * 1e6:.2f} µs")
