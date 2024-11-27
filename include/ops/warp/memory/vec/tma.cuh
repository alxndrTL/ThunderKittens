#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

#include <cuda.h>
#include <iostream>

namespace kittens {
namespace tma {

/* ----------   Prefetch Tensor Map  ---------- */

/**
 * @brief Prefetches data from global memory into a shared memory vector, along with the tensormap.
 *
 * @tparam SV A shared vector type with a TMA-compatible layout
 * @param[out] dst The destination shared memory vector.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in] vec_idx The coord of the requested vector.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void prefetch(SV &dst, const GL &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    for(int i = ::kittens::laneid(); i < detail::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * detail::sv_tma_dim1<SV>;

        asm volatile (
            "cp.async.bulk.prefetch.tensor.4d.L2.global.tile"
            " [%0, {%1, %2, %3, %4}];"
            :
            : "l"(tma_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    }
}

/* ----------   Async load and store data from gmem/smem  ---------- */

/**
 * @brief Asynchronously stores data into global memory from a shared memory vector.
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam SV A shared vector type with a TMA-compatible layout
 * @param[out] dst_tma_map The destination tensormap address in global memory
 * @param[in] src The source shared memory vector.
 * @param[in] vec_idx The coord of the vector destination.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_async(const GL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < detail::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * detail::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*detail::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        asm volatile (
            "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
            " [%0, {%2, %3, %4, %5}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    }
    store_commit_group();
}

/**
* @brief Asynchronously performs an add reduction and stores the result into global memory.
*
* This function performs an asynchronous add reduction operation using CUDA's cp.reduce.async.bulk.tensor instruction.
*
* @tparam SV A shared vector type with a TMA-compatible layout
* @param[out] dst_tma_map The destination tensormap address in global memory
* @param[in] src The source shared memory vector.
* @param[in] vec_idx The coord of the vector destination.
*/
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_add_async(const GL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < detail::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * detail::sv_tma_dim1<SV>;
        uint32_t src_tma_ptr = tma_ptr + i*detail::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        asm volatile (
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group"
            " [%0, {%2, %3, %4, %5}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_tma_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    }
    store_commit_group();
}

/**
* @brief Asynchronously performs an min reduction and stores the result into global memory.
*
* This function performs an asynchronous min reduction operation using CUDA's cp.reduce.async.bulk.tensor instruction.
*
* @tparam SV A shared vector type with a TMA-compatible layout
* @param[out] dst_tma_map The destination tensormap address in global memory
* @param[in] src The source shared memory vector.
* @param[in] vec_idx The coord of the vector destination.
*/
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_min_async(const GL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < detail::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * detail::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*detail::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        asm volatile (
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group"
            " [%0, {%2, %3, %4, %5}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    }
    store_commit_group();
}

/**
* @brief Asynchronously performs an max reduction and stores the result into global memory.
*
* This function performs an asynchronous max reduction operation using CUDA's cp.reduce.async.bulk.tensor instruction.
*
* @tparam SV A shared vector type with a TMA-compatible layout
* @param[out] dst_tma_map The destination tensormap address in global memory
* @param[in] src The source shared memory vector.
* @param[in] vec_idx The coord of the vector destination.
*/
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_max_async(const GL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = ::kittens::laneid(); i < detail::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * detail::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*detail::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        asm volatile (
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group"
            " [%0, {%2, %3, %4, %5}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    }
    store_commit_group();
}

/**
 * @brief Asynchronously loads data from global memory into a shared memory vector.
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam SV A shared vector type with a TMA-compatible layout
 * @param[out] dst The destination shared memory vector.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in] vec_idx The coord of the requested vector.
 * @param[in,out] bar The semaphore used for synchronization of the asynchronous copy.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore& bar) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    for(int i = ::kittens::laneid(); i < detail::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * detail::sv_tma_dim1<SV>;
        uint32_t dst_i_ptr = dst_ptr + i*detail::sv_tma_dim1<SV>*sizeof(typename SV::dtype);

        // printf("Thread %d: Issuing a TMA load of size %d bytes from %llu to %u on coordinates {%d, %d, %d, %d}\n",
        //        (int)(::kittens::laneid()),
        //        (int)(detail::sv_tma_dim1<SV>*sizeof(typename SV::dtype)),
        //        (uint64_t)tma_ptr,
        //        (uint32_t)dst_i_ptr,
        //        (int)tma_coord.c, (int)tma_coord.r, (int)tma_coord.d, (int)tma_coord.b);

        asm volatile (
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5, %6}], [%2];"
            :
            : "r"(dst_i_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    }
}

namespace cluster {

/**
 * @brief Asynchronously loads data from global memory into a shared memory vector, broadcast across a cluster
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam SV A shared vector type with a TMA-compatible layout
 * @param[out] dst The destination shared memory vector.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in,out] bar The semaphore used for synchronization of the asynchronous copy.
 * @param[in] vec_idx The coord of the requested vector.
 * @param[in] cluster_mask The mask of the clusters to broadcast to.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    for(int i = ::kittens::laneid(); i < detail::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * detail::sv_tma_dim1<SV>;
        uint32_t dst_i_ptr = dst_ptr + i*detail::sv_tma_dim1<SV>*sizeof(typename SV::dtype);

        asm volatile (
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
            " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
            :
            : "r"(dst_i_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "h"(cluster_mask)
            : "memory"
        );
    }
}


} // namespace cluster
} // namespace tma
} // namespace kittens