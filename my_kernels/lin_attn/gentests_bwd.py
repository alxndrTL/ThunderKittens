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
    A = torch.ones_like(v, dtype=torch.bfloat16, device='cuda').to(torch.float32)
elif TESTNAME in ['randn_all', 'randn_t0', 'randn_t1', 'randn_t0t1', 'randn_t2']:
    torch.random.manual_seed(42)
    q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
    A = torch.rand_like(v, dtype=torch.bfloat16, device='cuda').to(torch.float32)
else:
    print('Invalid test name')
    sys.exit(0)

o = pytorch_ref(q, k, v)
J = (o*A).sum()
J.backward(retain_graph=True)
grad_q, grad_k, grad_v = q.grad, k.grad, v.grad
grad_o = torch.autograd.grad(J, o, retain_graph=True)[0]

with open(f'bwd_{B}x{H}x{N}x{D}.txt', 'w') as f:
    qf = q.to(torch.float32).flatten().cpu().numpy().tolist()
    kf = k.to(torch.float32).flatten().cpu().numpy().tolist()
    vf = v.to(torch.float32).flatten().cpu().numpy().tolist()
    dof = grad_o.to(torch.float32).flatten().cpu().numpy().tolist()
    dqf_ref = grad_q.to(torch.float32).flatten().cpu().numpy().tolist()
    dkf_ref = grad_k.to(torch.float32).flatten().cpu().numpy().tolist()
    dvf_ref = grad_v.to(torch.float32).flatten().cpu().numpy().tolist()

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
        f.write(repr(dof[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(dqf_ref[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(dkf_ref[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(dvf_ref[i]))
        f.write(' ')
