#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <tuple>

#ifdef TORCH_COMPILE
#define TK_COMPILE_LIN_ATTN
#endif

//RTX4090
//D=16 => NUM_WORKERS 16 ACTIVE_TILES 8 is ok
//D=64 => NUM_WORKERS 8 ACTIVE_TILES 4 is ok
//D=128 => NUM_WORKERS 2 ACTIVE_TILES 1 is ok

#define NUM_WORKERS 8 //16
#define ACTIVE_TILES 4 //8
#define NUM_THREADS NUM_WORKERS*kittens::WARP_THREADS

#define ROWS 16
#define ATTN_D 64

using namespace kittens;

template<int WORKERS, kittens::ducks::st::all ST, int N_TILES>
__device__ inline void cumsum_inplace(ST (&x)[N_TILES], int total_block_idx) {
    constexpr int STRIDE = WORKERS*kittens::WARP_THREADS;

    for(int i = 1; i < N_TILES; i++) {
        #pragma unroll
        for(int j = threadIdx.x; j < ST::num_elements; j+=STRIDE) {
            x[(total_block_idx+i)%N_TILES].data[j] += x[(total_block_idx+i-1)%N_TILES].data[j];
        }
    }
}

template<int WORKERS, kittens::ducks::st::all ST, int N_TILES>
__device__ inline void revcumsum_inplace(ST (&x)[N_TILES], int total_block_idx) {
    constexpr int STRIDE = WORKERS*kittens::WARP_THREADS;

    for(int i = N_TILES-1; i > 0; i--) {
        #pragma unroll
        for(int j = threadIdx.x; j < ST::num_elements; j+=STRIDE) {
            x[(total_block_idx+i)%N_TILES].data[j] += x[(total_block_idx+i+1)%N_TILES].data[j];
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// ----------------------------------------- Forward kernel ------------------------------------------
// ---------------------------------------------------------------------------------------------------

struct fwd_globals {
    using q_tile = st_bf<ROWS, ATTN_D>;
    using k_tile = st_bf<ROWS, ATTN_D>;
    using v_tile = st_bf<ROWS, ATTN_D>;
    using o_tile = st_bf<ROWS, ATTN_D>;

    // global layouts
    using q_gl     = gl<bf16,  -1, -1, -1, ATTN_D, q_tile>;
    using k_gl     = gl<bf16,  -1, -1, -1, ATTN_D, k_tile>;
    using v_gl     = gl<bf16,  -1, -1, -1, ATTN_D, v_tile>;
    using o_gl     = gl<bf16,  -1, -1, -1, ATTN_D, o_tile>;

    // pointers
    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;

    long unsigned int n;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void linear_attention_fwd(const __grid_constant__ fwd_globals g) {

    const int batch = blockIdx.y;
    const int head  = blockIdx.x;

    int warpid = kittens::warpid(); 

    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);

    st_bf<ROWS, ATTN_D> (&qo_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>();
    st_bf<ROWS, ATTN_D> (&k_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>();
    st_bf<ROWS, ATTN_D> (&v_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>();
    st_bf<ATTN_D, ATTN_D> (&s_s)[ACTIVE_TILES + 1]  = al.allocate<st_bf<ATTN_D, ATTN_D>, ACTIVE_TILES + 1>();

    int total_block_idx = 0;

    if (warpid < ACTIVE_TILES + 1) {
        zero(s_s[warpid]);
    }

    int n_blocks = g.n / (ACTIVE_TILES * ROWS);

    for (int block = 0; block < n_blocks; block++) {
        rt_bf<ROWS, ATTN_D> q, k;
        rt_bf<ATTN_D, ROWS> kt;
        rt_bf<ROWS, ROWS> local_attn_bf;
        rt_fl<ROWS, ROWS> local_attn;
        rt_bf<ROWS, ATTN_D> v;
        rt_fl<ATTN_D, ATTN_D> accum;
        rt_fl<ROWS, ATTN_D> o;

        int cur_idx;
        if(warpid < ACTIVE_TILES) {
            cur_idx = block*ACTIVE_TILES + warpid;
            load(qo_s[warpid], g.q, {batch, head, cur_idx, 0});
            load(k_s[warpid], g.k, {batch, head, cur_idx, 0});
        }
        else {
            cur_idx = block*ACTIVE_TILES + warpid - ACTIVE_TILES;
            load(v_s[warpid-ACTIVE_TILES], g.v, {batch, head, cur_idx, 0});
        }
        __syncthreads();

        if(warpid < ACTIVE_TILES) {
            load(q, qo_s[warpid]);
            load(k, k_s[warpid]);

            zero(local_attn);
            mma_ABt(local_attn, q, k, local_attn);

            copy(local_attn_bf, local_attn);
            make_causal(local_attn_bf, local_attn_bf, kittens::base_types::constants<bf16>::zero());

            load(v, v_s[warpid]);
            auto &v_col = swap_layout_inplace(v); // could define v with col_l directly?

            zero(o);
            mma_AB(o, local_attn_bf, v_col, o);

            zero(accum);
            transpose_sep(kt, k);
            mma_AB(accum, kt, v_col, accum);
            store(s_s[(total_block_idx+warpid+1)%(ACTIVE_TILES+1)], accum);
        }

        __syncthreads();
        cumsum_inplace<NUM_WORKERS>(s_s, total_block_idx);
        __syncthreads();

        if(warpid < ACTIVE_TILES) {
            rt_bf<ATTN_D, ATTN_D> s;
            load(q, qo_s[warpid]); 
            load(s, s_s[(total_block_idx+warpid)%(ACTIVE_TILES+1)]); // could define s with col_l directly?
            auto &s_col = swap_layout_inplace(s);
            mma_AB(o, q, s_col, o); 
            store(qo_s[warpid], o);
        }
        total_block_idx = (total_block_idx+ACTIVE_TILES)%(ACTIVE_TILES+1); // count backwards on the ring
        __syncthreads();
        
        if(warpid < ACTIVE_TILES) {
            store(g.o, qo_s[warpid], {batch, head, cur_idx, 0});
        }
        __syncthreads();
    }
}

fwd_globals fwd_init(
    bf16 *d_q, bf16 *d_k, bf16 *d_v,
    bf16 *d_o,
    long unsigned int ATTN_B, long unsigned int ATTN_H, long unsigned int ATTN_N
) {
    // global pointers

    using globals = fwd_globals;

    using q_tile     = globals::q_tile;
    using k_tile     = globals::k_tile;
    using v_tile     = globals::v_tile;
    using o_tile     = globals::o_tile;

    // global layouts
    using q_gl     = globals::q_gl;
    using k_gl     = globals::k_gl;
    using v_gl     = globals::v_gl;
    using o_gl     = globals::o_gl;

    q_gl     q_arg{d_q, ATTN_B, ATTN_H, ATTN_N, nullptr};
    k_gl     k_arg{d_k, ATTN_B, ATTN_H, ATTN_N, nullptr};
    v_gl     v_arg{d_v, ATTN_B, ATTN_H, ATTN_N, nullptr};
    o_gl     o_arg{d_o, ATTN_B, ATTN_H, ATTN_N, nullptr};

    globals g{
        q_arg, k_arg, v_arg, o_arg, ATTN_N
    };
    return g;
}

// ---------------------------------------------------------------------------------------------------
// ----------------------------------------- Backward kernel -----------------------------------------
// ---------------------------------------------------------------------------------------------------

struct bwd_globals {
    using q_tile = st_bf<ROWS, ATTN_D>;
    using k_tile = st_bf<ROWS, ATTN_D>;
    using v_tile = st_bf<ROWS, ATTN_D>;
    using do_tile = st_bf<ROWS, ATTN_D>;

    // global layouts
    using q_gl     = gl<bf16, -1, -1, -1, ATTN_D, q_tile>;
    using k_gl     = gl<bf16, -1, -1, -1, ATTN_D, k_tile>;
    using v_gl     = gl<bf16, -1, -1, -1, ATTN_D, v_tile>;
    using do_gl    = gl<bf16, -1, -1, -1, ATTN_D, do_tile>;

    // pointers
    q_gl q;
    k_gl k;
    v_gl v;
    do_gl d_o;

    q_gl dq;
    k_gl dk;
    v_gl dv;

    long unsigned int n;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void linear_attention_bwd(const __grid_constant__ bwd_globals g) {

    const int batch = blockIdx.y;
    const int head  = blockIdx.x;

    int warpid = kittens::warpid(); 

    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);

    st_bf<ROWS, ATTN_D> (&dodqqdk_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>(); // do,dq for 1st loop, q,dk for 2nd loop
    st_bf<ROWS, ATTN_D> (&k_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>(); // k for 1st and 2nd
    st_bf<ROWS, ATTN_D> (&v_s)[ACTIVE_TILES]   = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>(); // v for 1st and 2nd
    st_bf<ATTN_D, ATTN_D> (&sds_s)[ACTIVE_TILES + 1]  = al.allocate<st_bf<ATTN_D, ATTN_D>, ACTIVE_TILES + 1>(); // s for 1st, ds for 2nd
    st_bf<ROWS, ATTN_D> (&dodv_s)[ACTIVE_TILES] = al.allocate<st_bf<ROWS, ATTN_D>, ACTIVE_TILES>(); // do,dv for 2nd
    
    // first loop : dq

    int total_block_idx = 0;

    if (warpid < ACTIVE_TILES + 1) {
        zero(sds_s[warpid]);
    }

    int n_blocks = g.n / (ACTIVE_TILES * ROWS);

    for (int block = 0; block < n_blocks; block++) {
        rt_bf<ROWS, ATTN_D> d_o, k, v;
        rt_bf<ATTN_D, ROWS> vt;
        rt_bf<ROWS, ROWS> local_attn_bf;
        rt_fl<ROWS, ROWS> local_attn;
        rt_fl<ATTN_D, ATTN_D> accum;
        rt_fl<ROWS, ATTN_D> dq;

        int cur_idx;
        if(warpid < ACTIVE_TILES) {
            cur_idx = block*ACTIVE_TILES + warpid;
            load(dodqqdk_s[warpid], g.d_o, {batch, head, cur_idx, 0});
            load(k_s[warpid], g.k, {batch, head, cur_idx, 0});
        }
        else {
            cur_idx = block*ACTIVE_TILES + warpid - ACTIVE_TILES;
            load(v_s[warpid-ACTIVE_TILES], g.v, {batch, head, cur_idx, 0});
        }
        __syncthreads();

        if(warpid < ACTIVE_TILES) {
            load(d_o, dodqqdk_s[warpid]);
            load(v, v_s[warpid]);
            
            zero(local_attn);
            mma_ABt(local_attn, d_o, v, local_attn);

            copy(local_attn_bf, local_attn);
            make_causal(local_attn_bf, local_attn_bf, kittens::base_types::constants<bf16>::zero());

            load(k, k_s[warpid]);
            auto &k_col = swap_layout_inplace(k); // could define k with col_l directly?

            zero(dq);
            mma_AB(dq, local_attn_bf, k_col, dq);

            transpose_sep(vt, v);
            zero(accum);
            mma_AB(accum, vt, k_col, accum);
            store(sds_s[(total_block_idx+warpid+1)%(ACTIVE_TILES+1)], accum);
        }

        __syncthreads();
        cumsum_inplace<NUM_WORKERS>(sds_s, total_block_idx);
        __syncthreads();

        if(warpid < ACTIVE_TILES) {
            rt_bf<ATTN_D, ATTN_D> s;
            load(d_o, dodqqdk_s[warpid]); //happens twice no???
            load(s, sds_s[(total_block_idx+warpid)%(ACTIVE_TILES+1)]); // could define s with col_l directly?
            auto &s_col = swap_layout_inplace(s);
            mma_AB(dq, d_o, s_col, dq); 
            store(dodqqdk_s[warpid], dq);
        }
        total_block_idx = (total_block_idx+ACTIVE_TILES)%(ACTIVE_TILES+1); // count backwards on the ring
        __syncthreads();
        
        if(warpid < ACTIVE_TILES) {
            store(g.dq, dodqqdk_s[warpid], {batch, head, cur_idx, 0});
        }
        __syncthreads();
    }

    // second loop : dk,dv

    total_block_idx = 0;

    if (warpid < ACTIVE_TILES + 1) {
        zero(sds_s[warpid]);
    }

    for (int block = n_blocks-1; block >= 0; block--) {
        rt_bf<ROWS, ATTN_D> d_o, q, k, v;
        rt_bf<ROWS, ATTN_D, col_l> q_col;
        rt_bf<ATTN_D, ROWS> qt;
        rt_bf<ROWS, ROWS> local_attn_bf;
        rt_fl<ROWS, ROWS> local_attn;
        rt_fl<ATTN_D, ATTN_D> accum;
        rt_fl<ROWS, ATTN_D> dk, dv;

        int cur_idx;
        if(warpid < ACTIVE_TILES) {
            cur_idx = block*ACTIVE_TILES + warpid;
            load(dodqqdk_s[warpid], g.q, {batch, head, cur_idx, 0});
            load(k_s[warpid], g.k, {batch, head, cur_idx, 0});
        }
        else {
            cur_idx = block*ACTIVE_TILES + warpid - ACTIVE_TILES;
            load(v_s[warpid-ACTIVE_TILES], g.v, {batch, head, cur_idx, 0});
            load(dodv_s[warpid-ACTIVE_TILES], g.d_o, {batch, head, cur_idx, 0});
        }
        __syncthreads();

        if(warpid < ACTIVE_TILES) {
            load(d_o, dodv_s[warpid]);

            // first part of dk
            load(v, v_s[warpid]);
            
            zero(local_attn);
            mma_ABt(local_attn, v, d_o, local_attn);

            copy(local_attn_bf, local_attn);
            //make_causal_t(local_attn_bf, local_attn_bf, kittens::base_types::constants<bf16>::zero());
            make_causal(local_attn_bf, local_attn_bf, kittens::base_types::constants<bf16>::zero());

            load(q, dodqqdk_s[warpid]);
            //auto &q_col = swap_layout_inplace(q); // could define q with col_l directly?
            swap_layout(q_col, q);

            zero(dk);
            mma_AB(dk, local_attn_bf, q_col, dk);

            // first part of dv
            load(k, k_s[warpid]);

            zero(local_attn);
            mma_ABt(local_attn, k, q, local_attn);

            copy(local_attn_bf, local_attn);
            //make_causal_t(local_attn_bf, local_attn_bf, kittens::base_types::constants<bf16>::zero());
            make_causal(local_attn_bf, local_attn_bf, kittens::base_types::constants<bf16>::zero());

            zero(dv);
            auto &d_o_col = swap_layout_inplace(d_o);
            mma_AB(dv, local_attn_bf, d_o_col, dv);

            // ds
            transpose_sep(qt, q); // todo : instead of transposing, use q_col defined above
            zero(accum);
            mma_AB(accum, qt, d_o_col, accum); // mma_AtB?
            store(sds_s[(total_block_idx+warpid+1)%(ACTIVE_TILES+1)], accum);
        }

        __syncthreads();
        //revcumsum_inplace<NUM_WORKERS>(sds_s, total_block_idx); // causes a divergence with python
        __syncthreads();

        if(warpid < ACTIVE_TILES) {
            rt_bf<ATTN_D, ATTN_D> ds;
            // second part of dk
            load(v, v_s[warpid]); //happens twice no???
            load(ds, sds_s[(total_block_idx+warpid+2)%(ACTIVE_TILES+1)]);
            mma_ABt(dk, v, ds, dk);
            store(dodqqdk_s[warpid], dk);
        } // TODO : split compute between warps

        __syncthreads();

        if(warpid < ACTIVE_TILES) {
            rt_bf<ATTN_D, ATTN_D> ds;
            // second part of dv
            load(k, k_s[warpid]); //happens twice no???
            load(ds, sds_s[(total_block_idx+warpid+2)%(ACTIVE_TILES+1)]); //happens twice no???
            auto &ds_col = swap_layout_inplace(ds);
            mma_AB(dv, k, ds_col, dv); 
            store(dodv_s[warpid], dv);
        } // TODO : split compute between warps

        total_block_idx = ((total_block_idx - ACTIVE_TILES) % (ACTIVE_TILES + 1) + (ACTIVE_TILES + 1)) % (ACTIVE_TILES + 1); // count forwards on the ring (and avoid negative modulo)
        __syncthreads();
        
        if(warpid < ACTIVE_TILES) {
            store(g.dk, dodqqdk_s[warpid], {batch, head, cur_idx, 0});
            store(g.dv, dodv_s[warpid], {batch, head, cur_idx, 0});
        } // TODO : split storing between warps
        __syncthreads();
    }
}

bwd_globals bwd_init(
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_do,
    bf16 *d_dq, bf16 *d_dk, bf16 *d_dv,
    long unsigned int ATTN_B, long unsigned int ATTN_H, long unsigned int ATTN_N
) {
    // global pointers

    using globals = bwd_globals;

    using q_tile     = globals::q_tile;
    using k_tile     = globals::k_tile;
    using v_tile     = globals::v_tile;
    using do_tile     = globals::do_tile;

    // global layouts
    globals::q_gl  q_arg{d_q, ATTN_B, ATTN_H, ATTN_N, nullptr};
    globals::k_gl  k_arg{d_k, ATTN_B, ATTN_H, ATTN_N, nullptr};
    globals::v_gl  v_arg{d_v, ATTN_B, ATTN_H, ATTN_N, nullptr};
    globals::do_gl do_arg{d_do, ATTN_B, ATTN_H, ATTN_N, nullptr};

    globals::q_gl dq_arg{d_dq, ATTN_B, ATTN_H, ATTN_N, nullptr};
    globals::k_gl dk_arg{d_dk, ATTN_B, ATTN_H, ATTN_N, nullptr};
    globals::v_gl dv_arg{d_dv, ATTN_B, ATTN_H, ATTN_N, nullptr};

    globals g{
        q_arg, k_arg, v_arg, do_arg, dq_arg, dk_arg, dv_arg, ATTN_N
    };
    return g;
}

#ifdef TK_COMPILE_LIN_ATTN
#include "pyutils/torch_helpers.cuh"
#include <iostream>
void dispatch_fwd( 
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o,
    int ATTN_B, int ATTN_H, int ATTN_N
){
    fwd_globals g = fwd_init(
        d_q, d_k, d_v,
        d_o,
        ATTN_B, ATTN_H, ATTN_N
    );

    // launch
    unsigned long mem_size = 100000; // 4090
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(
        linear_attention_fwd,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(ATTN_H, ATTN_B);
    linear_attention_fwd<<<grid,NUM_THREADS,mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

torch::Tensor lin_attn_forward(
    const torch::Tensor q, 
    const torch::Tensor k,
    const torch::Tensor v
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    int B = q.size(0);
    int H = q.size(1);
    int DV = v.size(3);
    int N  = q.size(2);
    int FD = k.size(3);

    // checks
    TORCH_CHECK(k.size(0) == B, "k batch?");
    TORCH_CHECK(k.size(1) == H, "k heads?");
    TORCH_CHECK(k.size(2) == N, "k length?");

    TORCH_CHECK(v.size(0) == B, "v batch?");
    TORCH_CHECK(v.size(1) == H, "v heads?");
    TORCH_CHECK(v.size(2) == N, "v length?");

    // allocate output
    torch::Tensor out = torch::empty({B, H, N, DV}, v.options());

    // convert to bf16
    c10::BFloat16 *q_bf16 = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_bf16 = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_bf16 = v.data_ptr<c10::BFloat16>();
    
    bf16 *d_q = reinterpret_cast<bf16*>(q_bf16);
    bf16 *d_k = reinterpret_cast<bf16*>(k_bf16);
    bf16 *d_v = reinterpret_cast<bf16*>(v_bf16);
    bf16 *d_o = reinterpret_cast<bf16*>(out.data_ptr<c10::BFloat16>());

    dispatch_fwd(
        d_q, d_k, d_v, d_o,
        B, H, N
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    return out;
    cudaDeviceSynchronize();
}

//todo : same for bwd

#else
#ifdef FWD_HARNESS
#include "4090_harness_fwd.impl"
#else
#include "4090_harness_bwd.impl"
#endif
#endif
