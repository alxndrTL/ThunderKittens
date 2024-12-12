#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <tuple>

#ifdef TORCH_COMPILE
#define TK_COMPILE_BASED
#endif

#define NUM_WORKERS 8 //16
#define ACTIVE_TILES 4 //8
#define NUM_THREADS NUM_WORKERS*kittens::WARP_THREADS
#define D_QK 16
#define D_VO 64

using namespace kittens;

struct based_globals { 
    using q_tile = st_bf<16, D_VO>;
    using k_tile = st_bf<16, D_VO>;
    using v_tile = st_bf<16, D_VO>;
    using o_tile = st_bf<16, D_VO>;

    // global layouts
    using q_gl     = gl<bf16,  -1, -1, -1, D_VO, q_tile>;
    using k_gl     = gl<bf16,  -1, -1, -1, D_VO, k_tile>;
    using v_gl     = gl<bf16,  -1, -1, -1, D_VO, v_tile>;
    using o_gl     = gl<bf16,  -1, -1, -1, D_VO, o_tile>;

    // pointers
    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;

    long unsigned int n;
};

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

__global__ __launch_bounds__(NUM_THREADS, 1)
void based_linear_attention(const __grid_constant__ based_globals g) {

    const int batch = blockIdx.y;
    const int head  = blockIdx.x;

    int warpid = kittens::warpid(); 

    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);

    st_bf<16,64> (&qo_s)[ACTIVE_TILES]   = al.allocate<st_bf<16,64>, ACTIVE_TILES>();
    st_bf<16,64> (&k_s)[ACTIVE_TILES]   = al.allocate<st_bf<16,64>, ACTIVE_TILES>();
    st_bf<16,64> (&v_s)[ACTIVE_TILES]   = al.allocate<st_bf<16,64>, ACTIVE_TILES>();
    st_bf<64,64> (&s_s)[ACTIVE_TILES + 1]  = al.allocate<st_bf<64, 64>, ACTIVE_TILES + 1>();

    int total_block_idx = 0;

    if (warpid < ACTIVE_TILES + 1) {
        zero(s_s[warpid]);
    }

    int n_blocks = g.n / (ACTIVE_TILES * kittens::TILE_ROW_DIM<bf16>);

    for (int block = 0; block < n_blocks; block++) {
        rt_bf<16, 64> q, k;
        rt_bf<64, 16> kt;
        rt_bf<16, 16> local_attn_bf;
        rt_fl<16, 16> local_attn;
        rt_bf<16, 64> v;
        rt_fl<64, 64> accum;
        rt_fl<16, 64> o;

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
            rt_bf<64, 64> s;
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

based_globals based_init(
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o,
    long unsigned int ATTN_B, long unsigned int ATTN_H, long unsigned int ATTN_N
) {
    // global pointers

    using globals = based_globals;

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

#ifdef TK_COMPILE_BASED
#include "pyutils/torch_helpers.cuh"
#include <iostream>
void dispatch_based( 
    bf16 *d_q, bf16 *d_k, bf16 *d_v, bf16 *d_o,
    int ATTN_B, int ATTN_H, int ATTN_N
){
    based_globals g = based_init(
        d_q, d_k, d_v, d_o,
        ATTN_B, ATTN_H, ATTN_N
    );

    // launch
    unsigned long mem_size = 100000; // 4090
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(
        based_linear_attention,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(ATTN_H, ATTN_B);
    based_linear_attention<<<grid,NUM_THREADS,mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

std::tuple<torch::Tensor, torch::Tensor> based(
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

    dispatch_based(
        d_q, d_k, d_v, d_o,
        B, H, N
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    return std::make_tuple(out);
    cudaDeviceSynchronize();
}
#else
#include "harness_4090.impl"
#endif
