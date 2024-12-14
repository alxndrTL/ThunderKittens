import sys
from tqdm import trange

import torch
from einops import rearrange

# TODO : q scale!

def pytorch_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale=False):
    if scale:
        q = q * (q.shape[-1] ** -0.5)
    attn = q @ k.transpose(-2, -1)
    attn.masked_fill_(~torch.tril(torch.ones(q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device)), 0)
    o = attn @ v
    return o

B = 1 # keep 1
H = 1 # keep 1
N = 1024
D = 64
scale = False

TESTNAME = sys.argv[1] if len(sys.argv) > 1 else 'randn_all'

if TESTNAME in ['ones_all', 'ones_t0', 'ones_t1', 'ones_t0t1', 'ones_t2']:
    q = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
    k = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
    v = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
elif TESTNAME in ['randn_all', 'randn_t0', 'randn_t1', 'randn_t0t1', 'randn_t2']:
    torch.random.manual_seed(42)
    q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
else:
    print('Invalid test name')
    sys.exit(0)

o = pytorch_ref(q, k, v)

with open(f'fwd_{B}x{H}x{N}x{D}.txt', 'w') as f:
    qf = q.to(torch.float32).flatten().cpu().numpy().tolist()
    kf = k.to(torch.float32).flatten().cpu().numpy().tolist()
    vf = v.to(torch.float32).flatten().cpu().numpy().tolist()
    of_ref = o.to(torch.float32).flatten().cpu().numpy().tolist()

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
        f.write(repr(of_ref[i]))
        f.write(' ')
