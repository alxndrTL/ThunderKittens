#include "kittens.cuh"

using namespace kittens;

constexpr int NUM_WORKERS = 6;
constexpr int PIPE_STAGES = 3; 

template<int D> constexpr size_t ROWS = 64; // height of each worker tile (rows) = chunk_size
template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, ROWS<D>, D, L>;
template<int D, typename T=float> using attn_tile = rt<T, ROWS<D>, ROWS<D>>;
template<int D, typename T=float, typename L=row_l> using h_tile = rt<T, D, D, L>;
template<int D> using shared_tile = st_bf<ROWS<D>, D>;
template<int D> using global_layout = gl<bf16, -1, -1, -1, D>; // B, N, H, specified at runtime, D known at compile time for this kernel
template<int D> struct globals { global_layout<D> Qg, Kg, Vg, Og; };

template<int D> __launch_bounds__(NUM_WORKERS*WARP_THREADS, 1)
__global__ void attend_ker(const __grid_constant__ globals<D> g) {

    using load_group = kittens::group<3>; // triplets of works collaboratively load q, k, v tiles
    int loadid = load_group::groupid(), workerid = kittens::warpid(); // which worker am I?
    constexpr int LOAD_BLOCKS = NUM_WORKERS / load_group::GROUP_WARPS;
    const int batch = blockIdx.y, head = blockIdx.x;

    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);
    shared_tile<D> (&q_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    shared_tile<D> (&k_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    shared_tile<D> (&v_smem)[LOAD_BLOCKS][PIPE_STAGES] = al.allocate<shared_tile<D>, LOAD_BLOCKS, PIPE_STAGES>();
    shared_tile<D> (&o_smem)[NUM_WORKERS] = reinterpret_cast<shared_tile<D>(&)[NUM_WORKERS]>(k_smem);

    // Initialize all of the register tiles.
    qkvo_tile<D, bf16> q_reg, k_reg; // Q and K are both row layout, as we use mma_ABt.
    qkvo_tile<D, bf16, col_l> v_reg; // V is column layout, as we use mma_AB.
    qkvo_tile<D, float> o_reg; // Output tile.
    attn_tile<D, float> att_block; // attention tile, in float. (We want to use float wherever possible.)
    attn_tile<D, bf16> att_block_mma; // bf16 attention tile for the second mma_AB. We cast right before that op.
    h_tile<D, bf16, col_l> h_reg; // mma_AB so colum layout!

    if constexpr(D == 16) mul(q_reg, q_reg, __float2bfloat16(0.25f));
    else if constexpr(D == 32) mul(q_reg, q_reg, __float2bfloat16(0.1767766953f));

    zero(o_reg);
    zero(h_reg);
    // launch the load of the first q, k, v tiles (cdiv(L, (number of groups) * ROWS) because each groups treats ROWS)
    int qkv_blocks = (g.Qg.depth + LOAD_BLOCKS*ROWS<D>-1) / (LOAD_BLOCKS*ROWS<D>);
    int tic = 0;
    //printf("%d\n", loadid);
    load_group::load_async<1, false>(q_smem[loadid][0], g.Qg, {batch, loadid, head, 0});
    load_group::load_async<1, false>(k_smem[loadid][0], g.Kg, {0, 0, 0, 0});
    load_group::load_async<1, false>(v_smem[loadid][0], g.Vg, {0, 0, 0, 0});

    //todo : other possible design choice
    //double for loop like in FA...
    
    load_group::load_async<1, false>(q_smem[loadid][0], g.Qg, {batch, loadid, head, 0});
    load_group::load_async<1, false>(k_smem[loadid][0], g.Kg, {batch, loadid, head, 0});
    load_group::load_async<1, false>(v_smem[loadid][0], g.Vg, {batch, loadid, head, 0});
    // iterate over k, v for these q's that have been loaded
    for(auto qkv_idx = 0; qkv_idx < qkv_blocks; qkv_idx++, tic=(tic+1)%PIPE_STAGES) {
        int next_load_idx = (qkv_idx+1)*LOAD_BLOCKS + loadid;
        if(next_load_idx*ROWS<D> < g.Qg.depth) {
            int next_tic = (tic+1)%PIPE_STAGES;
            load_group::load_async<1, false>(q_smem[loadid][next_tic], g.Qg, {batch, next_load_idx, head, 0});
            load_group::load_async<1, false>(k_smem[loadid][next_tic], g.Kg, {batch, next_load_idx, head, 0});
            load_group::load_async<1, false>(v_smem[loadid][next_tic], g.Vg, {batch, next_load_idx, head, 0});
            load_async_wait<1>(); // next q, k, v can stay in flight.
        }
        else load_async_wait();
        __syncthreads();

        #pragma unroll LOAD_BLOCKS
        for(int subtile = 0; subtile < LOAD_BLOCKS && (qkv_idx*LOAD_BLOCKS + subtile)*ROWS<D> < g.Qg.depth; subtile++) {
            load(q_reg, q_smem[subtile][tic]); // load q,k from shared into registers
            load(k_reg, k_smem[subtile][tic]);
            zero(att_block); // zero ROWSxROWS attention tile
            mma_ABt(att_block, q_reg, k_reg, att_block); // Q@K.T, (ROWS*ROWS)
            //todo : causal masking on att_block
            copy(att_block_mma, att_block); // todo: necessary?

            load(v_reg, v_smem[subtile][tic]);
            mma_AB(o_reg, att_block_mma, v_reg, o_reg); // (ROWS, D)
            mma_AB(o_reg, q_reg, h_reg, o_reg); // (ROWS, D)
            
            swap_layout_inplace(h_reg);
            
            mma_AtB(h_reg, k_reg, v_reg, h_reg); // (D, D)
        }
    }
}

#include "4090_harness.impl"