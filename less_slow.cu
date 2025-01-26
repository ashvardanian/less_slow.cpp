/**
 *  @brief  Low-level CUDA kernels for building a performance-first mindset.
 *  @file   less_slow.cuh
 *  @author Ash Vardanian
 *
 *  The contents of this file complement the contents of the `less_slow.cpp`
 *  file with GPGPU kernels showcasing:
 *
 *  - How to coordinate CUDA cores within a single block or warp?
 *    A.k.a. how to use shared memory, warp shuffle intrinsics, and reductions?
 *  - What are CUDA math intrinsics and how much faster are they?
 *    A.k.a. when to use `__sinf` over `sinf` or `__fdividef` over `a / b`?
 *  - What's the Physical Page Caching behavior on GPUs?
 *  - How to schedule advanced computational graphs on GPUs?
 *    A.k.a. CUDA streams vs Graph Node API vs Cooperative Groups?
 */
#pragma once
#include <cub/cub.cuh>
#include <cuda_runtime.h>

template <typename scalar_type_, std::size_t side_>
struct small_square_matrix {
    scalar_type_ scalars[side_][side_];
};

/**
 *  @brief  A CUDA kernel that computes the product of two small square matrices.
 *          Doesn't use any block/warp-level communication and optimizations.
 */
template <typename scalar_type_, std::size_t side_>
small_square_matrix<scalar_type_, side_> small_matmul_kernel_cuda( //
    small_square_matrix<scalar_type_, side_> const &a,             //
    small_square_matrix<scalar_type_, side_> const &b) {

    small_square_matrix<scalar_type_, side_> c;
    for (std::size_t i = 0; i != side_; ++i)
        for (std::size_t j = 0; j != side_; ++j)
            for (std::size_t k = 0; k != side_; ++k) c.scalars[i][j] += a.scalars[i][k] * b.scalars[k][j];
    return c;
}

#include <cuda_fp16.h> //`half` type
#include <mma.h>       // `mma::` intrinsics

__global__ void tops_f16_sm70tc_cuda_kernel() {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // To initialize, we can call: `wmma::fill_fragment(c_frag, 0.0f)`
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
}
