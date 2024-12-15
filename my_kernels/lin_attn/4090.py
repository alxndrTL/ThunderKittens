import sys
from tqdm import trange

import torch
from einops import rearrange

ROWS = 16
ATTN_D = 64
ACTIVE_TILES = 4
NUM_WORKERS = 8

def pytorch_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale=False):
    if scale:
        q = q * (q.shape[-1] ** -0.5)
    attn = q @ k.transpose(-2, -1)
    attn.masked_fill_(~torch.tril(torch.ones(q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device)), 0)
    o = attn @ v
    return o

def zero(x):
    x.zero_(0)

def make_causal(x):
    mask = torch.triu(torch.ones_like(x), diagonal=1)
    x.masked_fill_(mask.bool(), 0)

def make_causal_t(x): # TODO
    make_causal(x)

def mma_AB(out, A, B):
    torch.matmul(A, B, out=out)

def mma_ABt(out, A, B):
    torch.matmul(A, B.T, out=out)

def cumsum_inplace(sds_s, start_idx):
    accum = torch.zeros_like(sds_s[0])
    for i in range(ACTIVE_TILES + 1):
        accum += sds_s[(start_idx + i) % (ACTIVE_TILES + 1)]
        sds_s[(start_idx + i) % (ACTIVE_TILES + 1)] = accum.clone()

def revcumsum_inplace(sds_s, start_idx):
    accum = torch.zeros_like(sds_s[0])
    for i in range(ACTIVE_TILES, -1, -1):
        accum += sds_s[(start_idx + i) % (ACTIVE_TILES + 1)]
        sds_s[(start_idx + i) % (ACTIVE_TILES + 1)] = accum.clone()

def linear_attention_bwd(q, k, v, d_o, b, h):
    # q,k,v,d_o: (B,H,N,D)
    B,H,N,D = q.shape

    dq_g = torch.zeros_like(q)
    dk_g = torch.zeros_like(k)
    dv_g = torch.zeros_like(v)

    n_blocks = N // (ACTIVE_TILES * ROWS)

    dodqqdk_s = torch.zeros((ACTIVE_TILES, ROWS, D), dtype=q.dtype)
    k_s = torch.zeros((ACTIVE_TILES, ROWS, D), dtype=q.dtype)
    v_s = torch.zeros((ACTIVE_TILES, ROWS, D), dtype=q.dtype)
    sds_s = torch.zeros((ACTIVE_TILES+1, D, D), dtype=q.dtype)

    dq_r = torch.zeros((ACTIVE_TILES, ROWS, D), dtype=q.dtype)

    # ---- First loop: compute dq ----
    total_block_idx = 0
    for block in range(n_blocks):
        # Load tiles of d_o and k,v
        # We'll emulate tiled loading by slicing
        # cur_idx range: block*BLOCK to block*BLOCK + BLOCK

        for warpid in range(ACTIVE_TILES):
            cur_idx = block*ACTIVE_TILES + warpid
            dodqqdk_s[warpid] = d_o[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:]
            k_s[warpid] = k[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:]
        for warpid in range(ACTIVE_TILES, NUM_WORKERS):
            cur_idx = block*ACTIVE_TILES + warpid - ACTIVE_TILES
            v_s[warpid-ACTIVE_TILES] = v[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:]
        
        for warpid in range(ACTIVE_TILES):
            d_o_tile = dodqqdk_s[warpid]
            v_tile = v_s[warpid]

            local_attn = d_o_tile @ v_tile.T
            make_causal(local_attn)

            k_tile = k_s[warpid]

            dq_r[warpid] = local_attn @ k_tile

            accum = v_tile.T @ k_tile
            sds_s[(total_block_idx+warpid+1)%(ACTIVE_TILES+1)] = accum

        cumsum_inplace(sds_s, total_block_idx)

        for warpid in range(ACTIVE_TILES):
            d_o_tile = dodqqdk_s[warpid]
            s = sds_s[(total_block_idx+warpid)%(ACTIVE_TILES+1)]
            dq_r[warpid] += d_o_tile @ s
            dodqqdk_s[warpid] = dq_r[warpid]

        total_block_idx = (total_block_idx+ACTIVE_TILES)%(ACTIVE_TILES+1)

        for warpidx in range(ACTIVE_TILES):
            cur_idx = block*ACTIVE_TILES + warpidx
            dq_tile = dodqqdk_s[warpidx]
            dq_g[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:] = dq_tile

    """
    # ---- Second loop: compute dk, dv ----
    total_block_idx = 0
    zero(sds_s)
    for block in range(n_blocks-1, -1, -1):
        for tile in range(ACTIVE_TILES):
            cur_idx = block*ACTIVE_TILES + tile
            # Load q,k,v,d_o again
            # Similar logic to above, now for dk,dv
            q_slice = q[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:]
            k_slice = k[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:]
            v_slice = v[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:]
            d_o_slice = d_o[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:]

            # Compute dk,dv similarly:
            # local_attn = v_slice @ d_o_slice^T -> make_causal_t
            local_attn = v_slice @ d_o_slice.T
            make_causal_t(local_attn)
            dk_slice = local_attn @ q_slice  # partial dk

            # and similarly for dv:
            local_attn2 = k_slice @ q_slice.T
            make_causal_t(local_attn2)
            dv_slice = local_attn2 @ d_o_slice

            # ds for cumulative sums:
            ds = q_slice.T @ d_o_slice
            sds_s[(total_block_idx+tile+1)%(ACTIVE_TILES+1)] = ds

            dk[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:] += dk_slice
            dv[b,h,cur_idx*ROWS:(cur_idx+1)*ROWS,:] += dv_slice

        revcumsum_inplace(sds_s, total_block_idx)
        total_block_idx = ((total_block_idx - ACTIVE_TILES) % (ACTIVE_TILES + 1))
    """

    return dq_g, dk_g, dv_g


B = 1 # keep 1
H = 1 # keep 1
N = 1024
D = ATTN_D
scale = False

torch.random.manual_seed(42)
q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
A = torch.rand_like(v, dtype=torch.bfloat16, device='cuda').to(torch.float32)

q.requires_grad = True
k.requires_grad = True
v.requires_grad = True

o = pytorch_ref(q, k, v)
J = (o*A).sum()
J.backward(retain_graph=True)
grad_q, grad_k, grad_v = q.grad, k.grad, v.grad
grad_o = torch.autograd.grad(J, o, retain_graph=True)[0]

dq, dk, dv = linear_attention_bwd(q, k, v, grad_o, 0, 0)

print(grad_q.shape)
print(dq.shape)