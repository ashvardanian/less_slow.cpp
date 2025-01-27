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
 *
 *  To compile this file, dump the SASS code, and check for Tensor Cores usage:
 *
 *      nvcc -arch=sm_70 -Xptxas -v -lineinfo --extra-device-vectorization-info -cubin -o less_slow.cubin less_slow.cu
 *      cuobjdump -sass less_slow.cubin | grep -i mma
 */
#include <cuda_fp16.h> // `half` type

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

/**
 *  Starting with Nvidia Volta GPUs, specialized "Tensor Cores" @b (TC) are
 *  added for faster matrix multiplications. These Tensor Cores are much faster
 *  than native CUDA implementation of dot-product operations and provide
 *  special intrinsics for programmers to use.
 *
 *  Unlike typical CPU-side intrinsics, in CUDA, C++ templates are used.
 *  There is not a single Tensor Core generation that natively performs
 *  @b 16x16x16 FP16 matrix multiplication into FP32 accumulators.
 *  But we can use @b `wmma` 2D tiles of that size, that will be unpacked
 *  into the right combination of instructions at compile time.
 *
 *  Theoretically, this implies that we could have used 256x256x256 matrices,
 *  or some other size that optimally fits into the GPU's caches, shared along
 *  the cores in the same warp, but @b NO! Most sizes won't compile.
 *
 *  Moreover, splitting into hardware-specific tile sizes isn't done at the PTX
 *  level! It's done at the SASS level, so the PTX output for this kernel will
 *  still contain lines like:
 *
 *      wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {}, {}, {}, {};
 *
 *  That will be lowered to the right SASS instructions by the PTXAS assembler,
 *  and on Volta SM70 GPUs, will use the only supported size of 8x8x4:
 *
 *      HMMA.884.F32.F32.STEP2 R8, R2.reuse.ROW, R2.reuse.COL, R8
 *
 *  Unpacking it:
 *  - HMMA stands for Half-precision Matrix Multiply & Accumulate.
 *  - 884 stands for the 8x8x4 shape of the matrix multiplication.
 *  - F32.F32 defines the multiplication and accumulation precision.
 *  - STEPx denotes the stage of the computation for a specific tile, where
 *    each HMMA instruction contributes to completing a part of the final
 *    result. In our case we will get 4 STEPs, repeated 4 times, for a
 *    total of 16x HMMA instructions per WMMA intrinsic.
 *
 *  For optimal usage of Tensor Cores:
 *  - Ensure your matrix dimensions are multiples of the tile size (8x8x4 on Volta).
 *  - Use shared memory efficiently to reduce global memory accesses.
 *  - Properly align input and output matrices in memory (128-byte alignment).
 */
#include <mma.h> // `mma::` intrinsics

__global__ void tops_f16_sm70tc_cuda_kernel() {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // To initialize, we can call:
    wmma::fill_fragment(a_frag, __float2half_rn(1.0f));
    wmma::fill_fragment(b_frag, __float2half_rn(1.0f));
    wmma::fill_fragment(c_frag, 0.0f);

    // To better saturate the ALU, we could unroll a few iterations:
    for (int i = 0; i != 128; ++i) wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Impossible condition to prevent optimization
    if (threadIdx.x == -1) wmma::store_matrix_sync(nullptr, c_frag, 16, wmma::mem_row_major);
}
