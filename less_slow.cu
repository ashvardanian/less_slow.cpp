/**
 *  @brief  Low-level CUDA kernels for building a performance-first mindset.
 *  @file   less_slow.cuh
 *  @author Ash Vardanian
 *
 *  The contents of this file complement the contents of the `less_slow.cpp`
 *  file with GPGPU kernels showcasing:
 *
 *  - How to use Tensor Cores for matrix multiplications?
 *    What's the difference between `mma` and `wgmma` on Hopper?
 *  - TODO: How to coordinate CUDA cores within a single block or warp?
 *    A.k.a. how to use shared memory, warp shuffle intrinsics, and reductions?
 *  - TODO: What are CUDA math intrinsics and how much faster are they?
 *    A.k.a. when to use `__sinf` over `sinf` or `__fdividef` over `a / b`?
 *  - TODO: What's the Physical Page Caching behavior on GPUs?
 *  - TODO: How to schedule advanced computational graphs on GPUs?
 *    A.k.a. CUDA streams vs Graph Node API vs Cooperative Groups?
 *
 *  To compile this file, dump the SASS code, and check for Tensor Cores usage
 *  on Volta SM70 GPUs, use the following commands:
 *
 *  $ nvcc -arch=sm_90 -Xptxas -v -lineinfo -ptx -o less_slow_from_cu.ptx less_slow.cu
 *  $ nvcc -arch=sm_90 -Xptxas -v -lineinfo -cubin -o less_slow_from_cu.cubin less_slow.cu
 *  $ cuobjdump -sass less_slow_from_cu.cubin | grep -i mma
 *
 *  Keep in mind the following TC generations:
 *
 *  - Volta SM70: 1st generation of TCs, server V100 cards.
 *  - Turing SM75: 2nd generation of TCs, consumer RTX 30 cards.
 *  - Ampere SM80: 3rd generation of TCs, server A100 cards.
 *  - Ada Lovelace SM89: 4th generation of TCs, consumer RTX 40 cards.
 *  - Hopper SM90: 5th generation of TCs, server H100 cards.
 *
 *  Looking at server-side V100, A100, and H100 GPUs, most features are
 *  identical, except for @b shared-memory size and TCs:
 *
 *    Feature                              | V100     | A100     | H100
 *    -------------------------------------|----------|----------|----------
 *    Compute Capability                   | 7.0      | 8.0      | 9.0
 *    PTX Version                          | 6+       | 7+       | 8+
 *    CUDA Releases                        | 9-10     | 11+      | 12+
 *    -------------------------------------|----------|----------|----------
 *    Threads / Warp                       | 32       | 32       | 32
 *    Max Warps / SM                       | 64       | 64       | 64
 *    Max Threads / SM                     | 2048     | 2048     | 2048
 *    Max Thread Blocks (CTAs) / SM        | 32       | 32       | 32
 *    Max Thread Blocks / Thread Block Cl. | NA       | NA       | 16
 *    Max 32-bit Registers / SM            | 65536    | 65536    | 65536
 *    Max Registers / Thread Block (CTA)   | 65536    | 65536    | 65536
 *    Max Registers / Thread               | 255      | 255      | 255
 *    Max Thread Block Size (# of threads) | 1024     | 1024     | 1024
 *    -------------------------------------|----------|----------|----------
 *    Ratio of SM Registers to FP32 Cores  | 1024     | 1024     | 512
 *    Shared Memory Size / SM              | ≤ 96 KB  | ≤ 164 KB | ≤ 228 KB
 *    Tensor Core Generation               | 1st      | 3rd      | 5th
 *
 */
#include <cstdint> // `std::uint8_t`
#if (__CUDA_ARCH__ >= 700)
#include <cuda_fp16.h> // `half` type
#endif
#if (__CUDA_ARCH__ >= 750)
#include <cuda_bf16.h> // `__nv_bfloat16` type
#endif

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
 *  ! wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {}, {}, {}, {};
 *
 *  That will be lowered to the right SASS instructions by the PTXAS assembler,
 *  and on Volta SM70 GPUs, will use the only supported size of 8x8x4:
 *
 *  ! HMMA.884.F32.F32.STEP2 R8, R2.reuse.ROW, R2.reuse.COL, R8
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
 *
 *  @see Supported numeric types until Ampere SM80:
 *       https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#improved-tensor-core-operations
 *  @see "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture" paper
 *       from HKSTU: https://arxiv.org/pdf/2402.13499v1
 *
 */
#include <mma.h> // `mma::` intrinsics

/**
 *  @brief  A CUDA kernel that @b repeatedly computes the product of two small
 *          matrices of size MxN and NxK using Tensor Cores.
 */
template <typename input_type_, typename output_type_, int m_, int n_, int k_, int repetitions_>
__device__ inline void tops_tc_cuda_kernel() {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, m_, n_, k_, input_type_, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, m_, n_, k_, input_type_, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, m_, n_, k_, output_type_> c_frag;

    // To initialize, we can call:
    //
    //      wmma::fill_fragment(a_frag, 1);
    //      wmma::fill_fragment(b_frag, 1);
    //      wmma::fill_fragment(c_frag, 0);
    //
    // To better saturate the ALU, we could unroll a few iterations:
    for (int i = 0; i != repetitions_; ++i) wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Impossible condition to prevent optimization
    if (threadIdx.x == 2147483647) wmma::store_matrix_sync(nullptr, c_frag, 16, wmma::mem_row_major);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750) //? Binary Matrices require SM75 or higher

/**
 *  To process binary matrices we can't rely on addition and multiplication.
 *  A different set of mathematical operations is required, such as @b XOR or
 *  @b AND as multiplication and @b POPCOUNT as accumulation. The names of
 *  those operations are passed as extra arguments to the @b `bmma_sync`.
 *
 *  @see Docs: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sub-byte-operations
 */
template <typename input_type_, typename output_type_, int m_, int n_, int k_, int repetitions_>
__device__ inline void binary_tops_tc_cuda_kernel( //
    nvcuda::wmma::experimental::bmmaBitOp bit_op, nvcuda::wmma::experimental::bmmaAccumulateOp acc_op) {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, m_, n_, k_, input_type_, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, m_, n_, k_, input_type_, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, m_, n_, k_, output_type_> c_frag;
    for (int i = 0; i != repetitions_; ++i) wmma::bmma_sync(c_frag, a_frag, b_frag, c_frag, bit_op, acc_op);
    if (threadIdx.x == 2147483647) wmma::store_matrix_sync(nullptr, c_frag, 16, wmma::mem_row_major);
}

#endif

#pragma region Volta

__global__ void tops_f16f16_sm70tc_16x16x16_1024unroll_cuda_kernel() {
    //? On Volta: 8x8x4.
    //? On Turing: 8x8x4 / 16x8x8 / 16x8x16.
    //? On Ampere: 16x8x8 / 16x8x16.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    tops_tc_cuda_kernel<half, half, 16, 16, 16, 1024>();
#endif
}
__global__ void tops_f16f32_sm70tc_16x16x16_1024unroll_cuda_kernel() {
    //? On Volta: 8x8x4.
    //? On Turing: 8x8x4 / 16x8x8 / 16x8x16.
    //? On Ampere: 16x8x8 / 16x8x16.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    tops_tc_cuda_kernel<half, float, 16, 16, 16, 1024>();
#endif
}

#pragma endregion

#pragma region Turing

__global__ void tops_u8i32_sm75tc_16x16x16_1024unroll_cuda_kernel() {
    //? On Turing: 8x8x16.
    //? On Ampere: 8x8x16 / 16x8x16 / 16x8x32.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    tops_tc_cuda_kernel<std::uint8_t, int32_t, 16, 16, 16, 1024>();
#endif
}
__global__ void tops_u4i32_sm75tc_8x8x32_1024unroll_cuda_kernel() {
    //! The 16x16x16 won't compile, 8x8x32 will.
    //? On Turing: 8x8x32.
    //? On Ampere: 8x8x32 / 16x8x32 / 16x8x64.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    tops_tc_cuda_kernel<nvcuda::wmma::experimental::precision::u4, int32_t, 8, 8, 32, 1024>();
#endif
}
__global__ void tops_b1i32xor_sm75tc_8x8x128_1024unroll_cuda_kernel() {
    //! The 16x16x16 won't compile, 8x8x128 will.
    //? On Turing: 8x8x128.
    //? On Ampere: 8x8x128 / 16x8x128 / 16x8x256.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    binary_tops_tc_cuda_kernel<nvcuda::wmma::experimental::precision::b1, int32_t, 8, 8, 128, 1024>(
        nvcuda::wmma::experimental::bmmaBitOp::bmmaBitOpXOR,
        nvcuda::wmma::experimental::bmmaAccumulateOp::bmmaAccumulateOpPOPC);
#endif
}

#pragma endregion

#pragma region Ampere

__global__ void tops_bf16f32_sm80tc_16x16x16_1024unroll_cuda_kernel() {
    //? On Ampere: 16x8x8 / 16x8x16.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    tops_tc_cuda_kernel<__nv_bfloat16, float, 16, 16, 16, 1024>();
#endif
}
__global__ void tops_tf32f32_sm80tc_16x16x8_1024unroll_cuda_kernel() {
    //! The 16x16x16 won't compile, 16x16x8 will.
    //? On Ampere: 16x8x4.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    tops_tc_cuda_kernel<nvcuda::wmma::precision::tf32, float, 16, 16, 8, 1024>();
#endif
}
__global__ void tops_f64f64_sm80tc_8x8x4_1024unroll_cuda_kernel() {
    //! The 16x16x16 won't compile, 8x8x4 will.
    //? On Ampere: 8x8x4.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    tops_tc_cuda_kernel<double, double, 8, 8, 4, 1024>();
#endif
}

__global__ void tops_b1i32and_sm80tc_8x8x128_1024unroll_cuda_kernel() {
    //! The 16x16x16 won't compile, 8x8x128 will.
    //? On Ampere: 8x8x128 / 16x8x128 / 16x8x256.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    binary_tops_tc_cuda_kernel<nvcuda::wmma::experimental::precision::b1, int32_t, 8, 8, 128, 1024>(
        nvcuda::wmma::experimental::bmmaBitOp::bmmaBitOpAND,
        nvcuda::wmma::experimental::bmmaAccumulateOp::bmmaAccumulateOpPOPC);
#endif
}

#pragma endregion

/**
 *  MMA is not the only family of tensor core instructions:
 *
 *  - MMA for dense-dense synchronous matrix multiplication.
 *  - Sparse MMA for synchronous sparse-dense matrix multiplication with
 *    a known @b structured sparsity pattern. Those are handy when you have
 *    a portion X of Y consecutive cells equal to zero. X and Y are generally
 *    set to 2 and 4, respectively, for a "2:4" pattern.
 *  - @b WGMMA or Warp-Group MMA operates on 4 contiguous warps, forming 128
 *    contiguous threads, generalizing the original MMA in 2 ways:
 *
 *    1. They can be asynchronous, for more flexible scheduling.
 *    2. They can avoid accumulation, a.k.a $C = A * B$, not $C += A * B$.
 *
 *  The later are vastly more complex. Just compare our old MMA signature:
 *  ! {wmma.mma.sync.aligned}.{row.col}.{m16n16k16}.{f32.f32} { ........ }
 *  ? {        header       }.{ layout}.{  shape  }.{ types } { operands }
 *
 *  To the new WGMMA signature:
 *  ! {wgmma.mm_async.sync.aligned}.{m64n64k16}.{f32.f16.f16} { ........ },{ .... }
 *  ? {     much longer header    }.{  shape  }.{   types   } { operands },{ args }
 *
 *  @see "Fast Matrix-Multiplication with WGMMA on NVIDIA Hopper GPUs" by Colfax:
 *       https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/
 *  @see "Outperforming cuBLAS on H100: a Worklog" by Pranjal Shankhdhar:
 *       https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
 */