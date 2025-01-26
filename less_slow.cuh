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

namespace ashvardanian {
namespace less_slow {

template <typename scalar_type_, std::size_t side_>
struct small_square_matrix {
    scalar_type_ scalars[side_][side_];
};

/**
 *  @brief  A CUDA kernel that computes the product of two small square matrices.
 *          Doesn't use any block/warp-level communication and optimizations.
 */
template <typename scalar_type_, std::size_t side_>
__device__ small_square_matrix<scalar_type_, side_> small_matmul_kernel_cuda(
    small_square_matrix<scalar_type_, side_> const &a, small_square_matrix<scalar_type_, side_> const &b) noexcept {
    small_square_matrix<scalar_type_, side_> c;
    for (std::size_t i = 0; i != side_; ++i)
        for (std::size_t j = 0; j != side_; ++j)
            for (std::size_t k = 0; k != side_; ++k) c.scalars[i][j] += a.scalars[i][k] * b.scalars[k][j];
    return c;
}

} // namespace less_slow
} // namespace ashvardanian
