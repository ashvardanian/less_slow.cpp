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
 *  To compile this file, dump the SASS code, and check for Tensor Cores
 *  usage on Volta SM70 GPUs, use the following commands:
 *
 *  $ nvcc -arch=sm_90a -Xptxas -v -lineinfo -ptx -o less_slow_from_cu.ptx less_slow.cu
 *  $ nvcc -arch=sm_90a -Xptxas -v -lineinfo -cubin -o less_slow_from_cu.cubin less_slow.cu
 *  $ cuobjdump -sass less_slow_from_cu.cubin | grep -i mma
 *
 *  Assuming how aggressively NVCC unrolls loops and the number of kernels in
 *  this file, you may want to deduplicate them:
 *
 *  $ cuobjdump -sass less_slow_from_cu.cubin | grep -i mma | \
 *  $   sed -r 's/\/\*[^*]+\*\///g' | \
 *  $   sed -r 's/^[[:space:]]+//; s/[[:space:]]+$//' | \
 *  $   sort -u
 *
 *  Keep in mind the following TC generations:
 *
 *  - Volta SM70: 1st generation of TCs, server V100 cards.
 *  - Turing SM75: 2nd generation of TCs, consumer RTX 30 cards.
 *  - Ampere SM80: 3rd generation of TCs, server A100 cards.
 *  - Hopper SM90: 4th generation of TCs, server H100 cards.
 *  - Blackwell SM100: 5th generations of TCs, server B200 cards.
 *
 *  Looking at server-side V100, A100, and H100 GPUs, most features
 *  are identical, except for @b shared-memory size and TCs:
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
 *    Tensor Core Generation               | 1st      | 3rd      | 4th
 *
 */
#include <cstdint> // `std::uint8_t`

#if (__CUDA_ARCH__ >= 700)
#include <cuda_fp16.h> // `half` type
#endif
#if (__CUDA_ARCH__ >= 750)
#include <cuda_bf16.h> // `__nv_bfloat16` type
#endif
#if (__CUDA_ARCH__ >= 900)
#include <cuda_fp8.h> // `__nv_fp8*` types
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

#include <thrust/sort.h> // `thrust::sort`

void reverse_and_sort_with_thrust(std::uint32_t *device_pointer, std::size_t array_length) {
    // Assuming we don't use the `thrust::device_vector` iterators, we need to pass
    // the execution policy separately to enforce the GPU backend over the CPU.
    thrust::reverse(thrust::device, device_pointer, device_pointer + array_length);
    thrust::sort(thrust::device, device_pointer, device_pointer + array_length);
}

#include <cub/cub.cuh> // `cub::DeviceRadixSort`

std::size_t reverse_and_sort_with_cub_space(std::uint32_t *device_pointer, std::size_t array_length) {
    std::size_t temporary_bytes = 0;
    cub::DeviceRadixSort::SortKeys(     //
        NULL, temporary_bytes,          // temporary memory and its size
        device_pointer, device_pointer, // "in" and "out" arrays
        array_length                    // number of elements and optional parameters
    );
    return temporary_bytes;
}

void reverse_and_sort_with_cub(std::uint32_t *device_pointer, std::size_t array_length, void *temporary_pointer,
                               std::size_t temporary_bytes, cudaStream_t stream) {
    // CUB has no reversal kernel. So to schedule the Thrust and CUB operations
    // on the same CUDA `stream`, we need to wrap it into a "policy" object.
    auto policy = thrust::cuda::par.on(stream);

    thrust::reverse(policy, device_pointer, device_pointer + array_length);
    cub::DeviceRadixSort::SortKeys(          //
        temporary_pointer, temporary_bytes,  // temporary memory and its size
        device_pointer, device_pointer,      // "in" and "out" arrays pin to same memory
        array_length,                        // number of elements
        0, sizeof(std::uint32_t) * CHAR_BIT, // begin and end bit positions
        stream                               // CUDA stream
    );
}

#pragma region Numerics

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
 *  - @b HMMA stands for Half-precision Matrix Multiply & Accumulate.
 *  - @b 884 stands for the 8x8x4 shape of the matrix multiplication.
 *  - @b F32.F32 defines the multiplication and accumulation precision.
 *  - @b STEPx denotes the stage of the computation for a specific tile, where
 *    each HMMA instruction contributes to completing a part of the final
 *    result. In our case we will get 4 STEPs, repeated 4 times, for a
 *    total of 16x HMMA instructions per WMMA intrinsic.
 *
 *  For optimal usage of Tensor Cores:
 *  - Ensure your matrix dimensions are multiples of the tile size .
 *  - Use shared memory efficiently to reduce global memory accesses.
 *  - Properly align input and output matrices in memory to 128 bytes.
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
template <typename input_type_, typename output_type_, int m_, int n_, int k_, int repetitions_ = 128>
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

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750) //? Binary matrices require SM75 or higher

/**
 *  To process binary matrices we can't rely on addition and multiplication.
 *  A different set of mathematical operations is required, such as @b XOR or
 *  @b AND as multiplication and @b POPCOUNT as accumulation. The names of
 *  those operations are passed as extra arguments to the @b `bmma_sync`.
 *
 *  @see Docs: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sub-byte-operations
 */
template <typename input_type_, typename output_type_, int m_, int n_, int k_, int repetitions_ = 128>
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

__global__ void tops_f16f16_sm70wmma_16x16x16_loop128_cuda_kernel() {
    //? On Volta: 8x8x4.
    //? On Turing: 8x8x4 / 16x8x8 / 16x8x16.
    //? On Ampere: 16x8x8 / 16x8x16.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    tops_tc_cuda_kernel<half, half, 16, 16, 16>();
#endif
}
__global__ void tops_f16f32_sm70wmma_16x16x16_loop128_cuda_kernel() {
    //? On Volta: 8x8x4.
    //? On Turing: 8x8x4 / 16x8x8 / 16x8x16.
    //? On Ampere: 16x8x8 / 16x8x16.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    tops_tc_cuda_kernel<half, float, 16, 16, 16>();
#endif
}

#pragma endregion

#pragma region Turing

__global__ void tops_u8i32_sm75wmma_16x16x16_loop128_cuda_kernel() {
    //? On Turing: 8x8x16.
    //? On Ampere: 8x8x16 / 16x8x16 / 16x8x32.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    tops_tc_cuda_kernel<std::uint8_t, int32_t, 16, 16, 16>();
#endif
}
__global__ void tops_u4i32_sm75wmma_8x8x32_loop128_cuda_kernel() {
    //! The 16x16x16 won't compile, 8x8x32 will.
    //? On Turing: 8x8x32.
    //? On Ampere: 8x8x32 / 16x8x32 / 16x8x64.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    tops_tc_cuda_kernel<nvcuda::wmma::experimental::precision::u4, int32_t, 8, 8, 32>();
#endif
}
__global__ void tops_b1i32xor_sm75wmma_8x8x128_loop128_cuda_kernel() {
    //! The 16x16x16 won't compile, 8x8x128 will.
    //? On Turing: 8x8x128.
    //? On Ampere: 8x8x128 / 16x8x128 / 16x8x256.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    binary_tops_tc_cuda_kernel<nvcuda::wmma::experimental::precision::b1, int32_t, 8, 8, 128>(
        nvcuda::wmma::experimental::bmmaBitOp::bmmaBitOpXOR,
        nvcuda::wmma::experimental::bmmaAccumulateOp::bmmaAccumulateOpPOPC);
#endif
}

#pragma endregion

#pragma region Ampere

__global__ void tops_bf16f32_sm80wmma_16x16x16_loop128_cuda_kernel() {
    //? On Ampere: 16x8x8 / 16x8x16.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    tops_tc_cuda_kernel<__nv_bfloat16, float, 16, 16, 16>();
#endif
}
__global__ void tops_tf32f32_sm80wmma_16x16x8_loop128_cuda_kernel() {
    //! The 16x16x16 won't compile, 16x16x8 will.
    //? On Ampere: 16x8x4.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    tops_tc_cuda_kernel<nvcuda::wmma::precision::tf32, float, 16, 16, 8>();
#endif
}
__global__ void tops_f64f64_sm80wmma_8x8x4_loop128_cuda_kernel() {
    //! The 16x16x16 won't compile, 8x8x4 will.
    //? On Ampere: 8x8x4.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    tops_tc_cuda_kernel<double, double, 8, 8, 4>();
#endif
}

__global__ void tops_b1i32and_sm80wmma_8x8x128_loop128_cuda_kernel() {
    //! The 16x16x16 won't compile, 8x8x128 will.
    //? On Ampere: 8x8x128 / 16x8x128 / 16x8x256.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    binary_tops_tc_cuda_kernel<nvcuda::wmma::experimental::precision::b1, int32_t, 8, 8, 128>(
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
 *  - @b WGMMA or Warp-Group MMA generalizes the original MMA in 2 ways:
 *
 *    1. They can be asynchronous, for more flexible scheduling.
 *    2. They can avoid accumulation, a.k.a $C = A * B$, not $C += A * B$.
 *
 *  The WGMMA is vastly more complex.
 *
 *  Just compare our old MMA signature:
 *  ! {wmma.mma.sync.aligned}.{row.col}.{m16n16k16}.{f32.f32} { ........ }
 *  ? {        header       }.{ layout}.{  shape  }.{ types } { operands }
 *
 *  To the new WGMMA signature:
 *  ! {wgmma.mma_async.sync.aligned}.{m64n64k16}.{f32.f16.f16} { ........ },{ .... }
 *  ? {     much  longer header    }.{  shape  }.{   types   } { operands },{ args }
 *
 *  Not only the signature and "fragment" sizes differ, but also the scheduling
 *  approach has changed between Ampere and Hopper once again:
 *
 *  1. Pre-Volta fast kernels would individually invoke FMA instructions.
 *  2. Volta's HMMA instruction synchronizes a group of @b 8 threads called
 *     a @b "quadpair" (QP) to share data and perform an 8x8x4.
 *  3. Ampere synchronizes all the threads in the warp, typically @b 32.
 *  4. Hopper synchronizes 4 continuous warps, typically @b 128 threads.
 *
 *  Moreover, unlike the CPU, on the GPU, we can't expect the old instructions
 *  to perform well - there can be a significant performance penalty if you
 *  don't upgrade your PTX!
 *
 *  @see "Fast Matrix-Multiplication with WGMMA on NVIDIA Hopper GPUs" by Colfax:
 *       https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/
 *  @see "Outperforming cuBLAS on H100: a Worklog" by Pranjal Shankhdhar:
 *       https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
 *
 *  To make things worse, there are no `wgmma::` CUDA C++ intrinsics!
 *  The closest thing to them is the @b CuTe low-level collection of C++
 *  templates, wrapping raw PTX instructions into MMA @b "atoms".
 *  Just for Hopper alone, there is @b 10'000 lines of different supported
 *  shape instantiations in @b `mma_sm90.hpp`.
 *
 *  @see CUTLASS updates: https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md
 *  @see CUTLASS GEMM API: https://github.com/NVIDIA/cutlass/blob/main/media/docs/gemm_api.md
 *  @see "Deep Dive on CUTLASS Ping-Pong GEMM Kernel" by PyTorch:
 *       https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/
 *  @see Minimal SM90 WGMMA + TMA GEMM example in 100 lines in CUTLASS 3.5.1:
 *       https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/wgmma_sm90.cu
 *
 *  We can also write "inline PTX" in CUDA C++, the same way we can write
 *  "inline assembly" on the host side C++.
 *
 *  The instruction syntax for Warp-Group asynchronous instructions is very
 *  different, as at least one of the operand matrices has to be in shared
 *  memory (not registers). It's documented as in 2 variants:
 *
 *      wgmma.mma_async.sync.aligned.shape.dtype.tf32.tf32
 *          d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b;
 *      wgmma.mma_async.sync.aligned.shape.dtype.tf32.tf32
 *          d, a, b-desc, scale-d, imm-scale-a, imm-scale-b;
 *
 *  There is no "C" matrix involved at all, we are computing `D = A * B + D`.
 *  The `imm-scale` parameters can be used to either negate the inputs,
 *  or disable additive bias accumulation in the output. Both must be immediate
 *  values. The supported shapes list is also quite exhausting and differs for
 *  various numeric types. For half-precision floats:
 *
 *      .m64n8k8, .m64n16k8, .m64n24k8, .m64n32k8,
 *      .m64n40k8, .m64n48k8, .m64n56k8, .m64n64k8,
 *      .m64n72k8, .m64n80k8, .m64n88k8, .m64n96k8,
 *      .m64n104k8, .m64n112k8, .m64n120k8, .m64n128k8,
 *      .m64n136k8, .m64n144k8, .m64n152k8, .m64n160k8,
 *      .m64n168k8, .m64n176k8, .m64n184k8, .m64n192k8,
 *      .m64n200k8, .m64n208k8, .m64n216k8, .m64n224k8,
 *      .m64n232k8, .m64n240k8, .m64n248k8, .m64n256k8

 */
#pragma region Hopper

/**
 *  Ideally, both matrices A and B should be in shared memory. Both are
 *  defined using 64-bit descriptors with the following layout:
 *
 *      - 14 bits [0; 13]: start address
 *      - 14 bits [16; 29]: leading dimension byte offset
 *      - 14 bits [32; 45]: stride dimension byte offset
 *      - 3 bits [49; 51]: matrix base offset, valid only for "swizzling"
 *      - 2 bits [62; 63]: "swizzling" mode
 *
 *  The matrix layout in WGMMA can be normal or transposed, but its named
 *  differently. Non-Transposed for A and B is called K-Major. The Transposed
 *  variant is called M-Major for A and N-Major for B.
 *
 *  The matrices in the shared memory are made up of one or more "swizzle
 *  layout atom". The exact layout of these swizzle atoms depends on the
 *  swizzling mode, swizzle-atomicity, and the leading dimension.
 *
 *  Swizzling defines the order of the elements and can have 4 possible values:
 *
 *      0: no "swizzling" at all
 *      1: a 128-byte "swizzle" with a 1024 byte offset of a repeating pattern
 *      2: a 64-byte "swizzle" with a 512 byte offset of a repeating pattern
 *      3: a 32-byte "swizzle" with a 256 byte offset of a repeating pattern
 *
 *  Here is how that logic is packed together:
 */
__device__ std::uint64_t wgmma_descriptor(                                                //
    std::uint64_t address,                                                                //
    std::uint64_t leading_offset, std::uint64_t stride_offset, std::uint64_t base_offset, //
    std::uint64_t swizzle) {
    //! One of the most counter-intuitive things is how those matrix descriptors are composed.
    //! All fo the strides are in bytes, but divided by 16 (same as right-sift by four).
    return ((address & 0x3FFFF) >> 4) | ((leading_offset >> 4) << 16) | ((stride_offset >> 4) << 32) |
           (base_offset << 49) | (swizzle << 62);
}

__device__ void wgmma_f16f32_64x256x16(float r[128], std::uint64_t a_descriptor, std::uint64_t b_descriptor) {
    //! Interestingly, there are 2 variants of this instruction:
    //! 1. Both arguments are in shared memory, in which case 2 immediate values
    //!    can be used to transpose the inputs.
    //! 2. One argument is in shared memory, and the other one is in the registers,
    //!    in which case only one can be transposed, and only one immediate value
    //!    for that can be supplied!
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile( //
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, "
        "%56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, "
        "%72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, "
        "%88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103, "
        "%104, %105, %106, %107, %108, %109, %110, %111, "
        "%112, %113, %114, %115, %116, %117, %118, %119, "
        "%120, %121, %122, %123, %124, %125, %126, %127"
        "}, "
        "%128, %129, "
        "1, 1, 1, 0, 0;"
        : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7]), "=f"(r[8]),
          "=f"(r[9]), "=f"(r[10]), "=f"(r[11]), "=f"(r[12]), "=f"(r[13]), "=f"(r[14]), "=f"(r[15]), "=f"(r[16]),
          "=f"(r[17]), "=f"(r[18]), "=f"(r[19]), "=f"(r[20]), "=f"(r[21]), "=f"(r[22]), "=f"(r[23]), "=f"(r[24]),
          "=f"(r[25]), "=f"(r[26]), "=f"(r[27]), "=f"(r[28]), "=f"(r[29]), "=f"(r[30]), "=f"(r[31]), "=f"(r[32]),
          "=f"(r[33]), "=f"(r[34]), "=f"(r[35]), "=f"(r[36]), "=f"(r[37]), "=f"(r[38]), "=f"(r[39]), "=f"(r[40]),
          "=f"(r[41]), "=f"(r[42]), "=f"(r[43]), "=f"(r[44]), "=f"(r[45]), "=f"(r[46]), "=f"(r[47]), "=f"(r[48]),
          "=f"(r[49]), "=f"(r[50]), "=f"(r[51]), "=f"(r[52]), "=f"(r[53]), "=f"(r[54]), "=f"(r[55]), "=f"(r[56]),
          "=f"(r[57]), "=f"(r[58]), "=f"(r[59]), "=f"(r[60]), "=f"(r[61]), "=f"(r[62]), "=f"(r[63]), "=f"(r[64]),
          "=f"(r[65]), "=f"(r[66]), "=f"(r[67]), "=f"(r[68]), "=f"(r[69]), "=f"(r[70]), "=f"(r[71]), "=f"(r[72]),
          "=f"(r[73]), "=f"(r[74]), "=f"(r[75]), "=f"(r[76]), "=f"(r[77]), "=f"(r[78]), "=f"(r[79]), "=f"(r[80]),
          "=f"(r[81]), "=f"(r[82]), "=f"(r[83]), "=f"(r[84]), "=f"(r[85]), "=f"(r[86]), "=f"(r[87]), "=f"(r[88]),
          "=f"(r[89]), "=f"(r[90]), "=f"(r[91]), "=f"(r[92]), "=f"(r[93]), "=f"(r[94]), "=f"(r[95]), "=f"(r[96]),
          "=f"(r[97]), "=f"(r[98]), "=f"(r[99]), "=f"(r[100]), "=f"(r[101]), "=f"(r[102]), "=f"(r[103]), "=f"(r[104]),
          "=f"(r[105]), "=f"(r[106]), "=f"(r[107]), "=f"(r[108]), "=f"(r[109]), "=f"(r[110]), "=f"(r[111]),
          "=f"(r[112]), "=f"(r[113]), "=f"(r[114]), "=f"(r[115]), "=f"(r[116]), "=f"(r[117]), "=f"(r[118]),
          "=f"(r[119]), "=f"(r[120]), "=f"(r[121]), "=f"(r[122]), "=f"(r[123]), "=f"(r[124]), "=f"(r[125]),
          "=f"(r[126]), "=f"(r[127])
        : "l"(a_descriptor), "l"(b_descriptor));
#endif
}

__device__ void wgmma_bf16f32_64x256x16(float r[128], std::uint64_t a_descriptor, std::uint64_t b_descriptor) {
    // The `bf16` instructions are almost identical to `f16`.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile( //
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, "
        "%56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, "
        "%72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, "
        "%88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103, "
        "%104, %105, %106, %107, %108, %109, %110, %111, "
        "%112, %113, %114, %115, %116, %117, %118, %119, "
        "%120, %121, %122, %123, %124, %125, %126, %127"
        "}, "
        "%128, %129, "
        "1, 1, 1, 0, 0;"
        : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7]), "=f"(r[8]),
          "=f"(r[9]), "=f"(r[10]), "=f"(r[11]), "=f"(r[12]), "=f"(r[13]), "=f"(r[14]), "=f"(r[15]), "=f"(r[16]),
          "=f"(r[17]), "=f"(r[18]), "=f"(r[19]), "=f"(r[20]), "=f"(r[21]), "=f"(r[22]), "=f"(r[23]), "=f"(r[24]),
          "=f"(r[25]), "=f"(r[26]), "=f"(r[27]), "=f"(r[28]), "=f"(r[29]), "=f"(r[30]), "=f"(r[31]), "=f"(r[32]),
          "=f"(r[33]), "=f"(r[34]), "=f"(r[35]), "=f"(r[36]), "=f"(r[37]), "=f"(r[38]), "=f"(r[39]), "=f"(r[40]),
          "=f"(r[41]), "=f"(r[42]), "=f"(r[43]), "=f"(r[44]), "=f"(r[45]), "=f"(r[46]), "=f"(r[47]), "=f"(r[48]),
          "=f"(r[49]), "=f"(r[50]), "=f"(r[51]), "=f"(r[52]), "=f"(r[53]), "=f"(r[54]), "=f"(r[55]), "=f"(r[56]),
          "=f"(r[57]), "=f"(r[58]), "=f"(r[59]), "=f"(r[60]), "=f"(r[61]), "=f"(r[62]), "=f"(r[63]), "=f"(r[64]),
          "=f"(r[65]), "=f"(r[66]), "=f"(r[67]), "=f"(r[68]), "=f"(r[69]), "=f"(r[70]), "=f"(r[71]), "=f"(r[72]),
          "=f"(r[73]), "=f"(r[74]), "=f"(r[75]), "=f"(r[76]), "=f"(r[77]), "=f"(r[78]), "=f"(r[79]), "=f"(r[80]),
          "=f"(r[81]), "=f"(r[82]), "=f"(r[83]), "=f"(r[84]), "=f"(r[85]), "=f"(r[86]), "=f"(r[87]), "=f"(r[88]),
          "=f"(r[89]), "=f"(r[90]), "=f"(r[91]), "=f"(r[92]), "=f"(r[93]), "=f"(r[94]), "=f"(r[95]), "=f"(r[96]),
          "=f"(r[97]), "=f"(r[98]), "=f"(r[99]), "=f"(r[100]), "=f"(r[101]), "=f"(r[102]), "=f"(r[103]), "=f"(r[104]),
          "=f"(r[105]), "=f"(r[106]), "=f"(r[107]), "=f"(r[108]), "=f"(r[109]), "=f"(r[110]), "=f"(r[111]),
          "=f"(r[112]), "=f"(r[113]), "=f"(r[114]), "=f"(r[115]), "=f"(r[116]), "=f"(r[117]), "=f"(r[118]),
          "=f"(r[119]), "=f"(r[120]), "=f"(r[121]), "=f"(r[122]), "=f"(r[123]), "=f"(r[124]), "=f"(r[125]),
          "=f"(r[126]), "=f"(r[127])
        : "l"(a_descriptor), "l"(b_descriptor));
#endif
}

__device__ void wgmma_tf32f32_64x256x8(float r[128], std::uint64_t a_descriptor, std::uint64_t b_descriptor) {
    //! Unlike the `f16` and `bf16` instructions, the `tf32` has fewer operands,
    //! and can't transpose the input matrices!
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile( //
        "wgmma.mma_async.sync.aligned.m64n256k8.f32.tf32.tf32 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, "
        "%56, %57, %58, %59, %60, %61, %62, %63, "
        "%64, %65, %66, %67, %68, %69, %70, %71, "
        "%72, %73, %74, %75, %76, %77, %78, %79, "
        "%80, %81, %82, %83, %84, %85, %86, %87, "
        "%88, %89, %90, %91, %92, %93, %94, %95, "
        "%96, %97, %98, %99, %100, %101, %102, %103, "
        "%104, %105, %106, %107, %108, %109, %110, %111, "
        "%112, %113, %114, %115, %116, %117, %118, %119, "
        "%120, %121, %122, %123, %124, %125, %126, %127"
        "}, "
        "%128, %129, "
        "1, 1, 1;"
        : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7]), "=f"(r[8]),
          "=f"(r[9]), "=f"(r[10]), "=f"(r[11]), "=f"(r[12]), "=f"(r[13]), "=f"(r[14]), "=f"(r[15]), "=f"(r[16]),
          "=f"(r[17]), "=f"(r[18]), "=f"(r[19]), "=f"(r[20]), "=f"(r[21]), "=f"(r[22]), "=f"(r[23]), "=f"(r[24]),
          "=f"(r[25]), "=f"(r[26]), "=f"(r[27]), "=f"(r[28]), "=f"(r[29]), "=f"(r[30]), "=f"(r[31]), "=f"(r[32]),
          "=f"(r[33]), "=f"(r[34]), "=f"(r[35]), "=f"(r[36]), "=f"(r[37]), "=f"(r[38]), "=f"(r[39]), "=f"(r[40]),
          "=f"(r[41]), "=f"(r[42]), "=f"(r[43]), "=f"(r[44]), "=f"(r[45]), "=f"(r[46]), "=f"(r[47]), "=f"(r[48]),
          "=f"(r[49]), "=f"(r[50]), "=f"(r[51]), "=f"(r[52]), "=f"(r[53]), "=f"(r[54]), "=f"(r[55]), "=f"(r[56]),
          "=f"(r[57]), "=f"(r[58]), "=f"(r[59]), "=f"(r[60]), "=f"(r[61]), "=f"(r[62]), "=f"(r[63]), "=f"(r[64]),
          "=f"(r[65]), "=f"(r[66]), "=f"(r[67]), "=f"(r[68]), "=f"(r[69]), "=f"(r[70]), "=f"(r[71]), "=f"(r[72]),
          "=f"(r[73]), "=f"(r[74]), "=f"(r[75]), "=f"(r[76]), "=f"(r[77]), "=f"(r[78]), "=f"(r[79]), "=f"(r[80]),
          "=f"(r[81]), "=f"(r[82]), "=f"(r[83]), "=f"(r[84]), "=f"(r[85]), "=f"(r[86]), "=f"(r[87]), "=f"(r[88]),
          "=f"(r[89]), "=f"(r[90]), "=f"(r[91]), "=f"(r[92]), "=f"(r[93]), "=f"(r[94]), "=f"(r[95]), "=f"(r[96]),
          "=f"(r[97]), "=f"(r[98]), "=f"(r[99]), "=f"(r[100]), "=f"(r[101]), "=f"(r[102]), "=f"(r[103]), "=f"(r[104]),
          "=f"(r[105]), "=f"(r[106]), "=f"(r[107]), "=f"(r[108]), "=f"(r[109]), "=f"(r[110]), "=f"(r[111]),
          "=f"(r[112]), "=f"(r[113]), "=f"(r[114]), "=f"(r[115]), "=f"(r[116]), "=f"(r[117]), "=f"(r[118]),
          "=f"(r[119]), "=f"(r[120]), "=f"(r[121]), "=f"(r[122]), "=f"(r[123]), "=f"(r[124]), "=f"(r[125]),
          "=f"(r[126]), "=f"(r[127])
        : "l"(a_descriptor), "l"(b_descriptor));
#endif
}

__device__ void wgmma_fence() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("wgmma.fence.sync.aligned;");
#endif
}

__device__ void wgmma_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("wgmma.commit_group.sync.aligned;");
#endif
}

__device__ void wgmma_sync_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("wgmma.wait_group.sync.aligned 0;");
#endif
}

__global__ void tops_f16f32_sm90wgmma_64x256x16_loop128_cuda_kernel() {
    // 64x256x16 is the largest tile size for `f16` supported on Hopper.
    // We can use `half` for type, but `uint16_t` is more portable.
    __shared__ std::uint16_t a_shared[64][16];
    __shared__ std::uint16_t b_shared[256][16];

    float c_registers[128] = {0.0f};
    std::uint64_t a_descriptor = wgmma_descriptor((std::uint64_t)a_shared, 128, 256, 0, 0);
    std::uint64_t b_descriptor = wgmma_descriptor((std::uint64_t)b_shared, 128 * 256 / 8, 128, 0, 0);
    wgmma_fence();
    for (int i = 0; i != 128; ++i) {
        wgmma_f16f32_64x256x16(c_registers, a_descriptor, b_descriptor);
        wgmma_commit_group();
    }
    wgmma_sync_group();
    if (threadIdx.x == 2147483647) *(std::uint16_t *)nullptr = c_registers[0];
}

__global__ void tops_bf16f32_sm90wgmma_64x256x16_loop128_cuda_kernel() {
    // 64x256x16 is the largest tile size for `bf16` supported on Hopper.
    // We can use `__nv_bfloat16` for type, but `uint16_t` is more portable.
    __shared__ std::uint16_t a_shared[64][16];
    __shared__ std::uint16_t b_shared[256][16];

    float c_registers[128] = {0.0f};
    std::uint64_t a_descriptor = wgmma_descriptor((std::uint64_t)a_shared, 128, 256, 0, 0);
    std::uint64_t b_descriptor = wgmma_descriptor((std::uint64_t)b_shared, 128 * 256 / 8, 128, 0, 0);
    wgmma_fence();
    for (int i = 0; i != 128; ++i) {
        wgmma_bf16f32_64x256x16(c_registers, a_descriptor, b_descriptor);
        wgmma_commit_group();
    }
    wgmma_sync_group();
    if (threadIdx.x == 2147483647) *(std::uint16_t *)nullptr = c_registers[0];
}

__global__ void tops_tf32f32_sm90wgmma_64x256x8_loop128_cuda_kernel() {
    // 64x256x8 is the largest tile size for `tf32` supported on Hopper.
    // Four-byte representations should be used for storage. Each entry will
    // shifted right by 13 bits before multiplication.
    __shared__ std::uint32_t a_shared[64][8];
    __shared__ std::uint32_t b_shared[256][8];

    // TODO: Unlike smaller 2-byte floats, the stride sizes will be different here.
    float c_registers[128] = {0.0f};
    std::uint64_t a_descriptor = wgmma_descriptor((std::uint64_t)a_shared, 128, 256, 0, 0);
    std::uint64_t b_descriptor = wgmma_descriptor((std::uint64_t)b_shared, 128 * 256 / 8, 128, 0, 0);
    wgmma_fence();
    for (int i = 0; i != 128; ++i) {
        wgmma_tf32f32_64x256x8(c_registers, a_descriptor, b_descriptor);
        wgmma_commit_group();
    }
    wgmma_sync_group();
    if (threadIdx.x == 2147483647) *(std::uint32_t *)nullptr = c_registers[0];
}

#pragma endregion

/**
 *
 *  @see "Blackwell Cluster Launch Control" in CUTLASS docs:
 *       https://github.com/NVIDIA/cutlass/blob/main/media/docs/blackwell_cluster_launch_control.md
 *
 */