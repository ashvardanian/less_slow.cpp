# ----------------------------------------------------------------------------
# less_slow_amd64.asm
# Micro-kernels for building a performance-first mindset for 64-bit x86.
# ----------------------------------------------------------------------------
# Export symbols so linkers can find them:

    .section .text
    .global i32_add_asm_kernel

    .global tops_f64_avx512_asm_kernel
    .global tops_f32_avx512_asm_kernel
    .global tops_f16_avx512_asm_kernel
    .global tops_bf16_avx512_asm_kernel
    .global tops_i16_avx512_asm_kernel
    .global tops_u8i8_avx512_asm_kernel

    .global tops_f64_avx2_asm_kernel
    .global tops_f32_avx2_asm_kernel

    .global tops_bf16_amx_asm_kernel
    .global tops_u8_amx_asm_kernel
    .global tops_i8_amx_asm_kernel

# ----------------------------------------------------------------------------
# Simple function that adds two 32-bit signed integers using System V AMD64.
# Arguments in 32-bit registers EDI (a) and ESI (b). Return value in EAX.
# ----------------------------------------------------------------------------
i32_add_asm_kernel:
    addl    %esi, %edi      # EDI = EDI + ESI
    movl    %edi, %eax      # EAX = EDI
    ret

# ----------------------------------------------------------------------------
# AVX-512 micro-kernels measuring Tensor Operations Per Second (TOPS)
# without accounting for memory bandwidth or latency. The following kernels
# assume presence of 32x 512-bit long registers (ZMM0-ZMM31).
# Each reports return the aggregate number of floating-point operations - 
# multiplications and additions forming the FMA instruction - performed.
# ----------------------------------------------------------------------------

tops_f64_avx512_asm_kernel:
    # Each vfmadd231pd does: DEST = DEST + (SRC1 * SRC2)
    # That is 8 multiplies + 8 adds = 16 FLOPs per instruction.
    # We'll do 10 instructions below => 10 × 16 = 160 FLOPs total.
    vfmadd231pd %zmm1, %zmm2, %zmm0     # 1
    vfmadd231pd %zmm4, %zmm5, %zmm3     # 2
    vfmadd231pd %zmm7, %zmm8, %zmm6     # 3
    vfmadd231pd %zmm10, %zmm11, %zmm9   # 4
    vfmadd231pd %zmm13, %zmm14, %zmm12  # 5
    vfmadd231pd %zmm16, %zmm17, %zmm15  # 6
    vfmadd231pd %zmm19, %zmm20, %zmm18  # 7
    vfmadd231pd %zmm22, %zmm23, %zmm21  # 8
    vfmadd231pd %zmm25, %zmm26, %zmm24  # 9
    vfmadd231pd %zmm28, %zmm29, %zmm27  # 10

    # Return value in 64-bit register %rax => 160 TOPs.
    movabsq $160, %rax
    ret

tops_f32_avx512_asm_kernel:
    vfmadd231ps %zmm1, %zmm2, %zmm0    # 1
    vfmadd231ps %zmm4, %zmm5, %zmm3    # 2
    vfmadd231ps %zmm7, %zmm8, %zmm6    # 3
    vfmadd231ps %zmm10, %zmm11, %zmm9    # 4
    vfmadd231ps %zmm13, %zmm14, %zmm12   # 5
    vfmadd231ps %zmm16, %zmm17, %zmm15   # 6
    vfmadd231ps %zmm19, %zmm20, %zmm18   # 7
    vfmadd231ps %zmm22, %zmm23, %zmm21   # 8
    vfmadd231ps %zmm25, %zmm26, %zmm24   # 9
    vfmadd231ps %zmm28, %zmm29, %zmm27   # 10
    movabsq $320, %rax
    ret

tops_f16_avx512_asm_kernel:
    # This kernel requires AVX512_FP16 support.
    vfmadd231ph %zmm1, %zmm2, %zmm0
    vfmadd231ph %zmm4, %zmm5, %zmm3
    vfmadd231ph %zmm7, %zmm8, %zmm6
    vfmadd231ph %zmm10, %zmm11, %zmm9
    vfmadd231ph %zmm13, %zmm14, %zmm12
    vfmadd231ph %zmm16, %zmm17, %zmm15
    vfmadd231ph %zmm19, %zmm20, %zmm18
    vfmadd231ph %zmm22, %zmm23, %zmm21
    vfmadd231ph %zmm25, %zmm26, %zmm24
    vfmadd231ph %zmm28, %zmm29, %zmm27
    movabsq $640, %rax
    ret

tops_bf16_avx512_asm_kernel:
    # This kernel requires AVX512_BF16 support.
    vdpbf16ps %zmm0, %zmm1, %zmm2
    vdpbf16ps %zmm3, %zmm4, %zmm5
    vdpbf16ps %zmm6, %zmm7, %zmm8
    vdpbf16ps %zmm9, %zmm10, %zmm11
    vdpbf16ps %zmm12, %zmm13, %zmm14
    vdpbf16ps %zmm15, %zmm16, %zmm17
    vdpbf16ps %zmm18, %zmm19, %zmm20
    vdpbf16ps %zmm21, %zmm22, %zmm23
    vdpbf16ps %zmm24, %zmm25, %zmm26
    vdpbf16ps %zmm27, %zmm28, %zmm29
    movabsq $640, %rax
    ret

tops_i16_avx512_asm_kernel:
    # This kernel requires AVX512_VNNI support.
    vpdpwssds %zmm0, %zmm1, %zmm2
    vpdpwssds %zmm3, %zmm4, %zmm5
    vpdpwssds %zmm6, %zmm7, %zmm8
    vpdpwssds %zmm9, %zmm10, %zmm11
    vpdpwssds %zmm12, %zmm13, %zmm14
    vpdpwssds %zmm15, %zmm16, %zmm17
    vpdpwssds %zmm18, %zmm19, %zmm20
    vpdpwssds %zmm21, %zmm22, %zmm23
    vpdpwssds %zmm24, %zmm25, %zmm26
    vpdpwssds %zmm27, %zmm28, %zmm29
    movabsq $640, %rax
    ret


tops_u8i8_avx512_asm_kernel:
    # This kernel requires AVX512_VNNI support.
    # This instruction is tricky! It's not strictly `u8` by `u8` to `u32`
    # multiplication, but rather `u8` by `i8` to `i32`! There is a similar
    # `vpdpbusds` instruction, that only differs by its saturation behavior.
    vpdpbusd %zmm0, %zmm1, %zmm2
    vpdpbusd %zmm3, %zmm4, %zmm5
    vpdpbusd %zmm6, %zmm7, %zmm8
    vpdpbusd %zmm9, %zmm10, %zmm11
    vpdpbusd %zmm12, %zmm13, %zmm14
    vpdpbusd %zmm15, %zmm16, %zmm17
    vpdpbusd %zmm18, %zmm19, %zmm20
    vpdpbusd %zmm21, %zmm22, %zmm23
    vpdpbusd %zmm24, %zmm25, %zmm26
    vpdpbusd %zmm27, %zmm28, %zmm29
    movabsq $1280, %rax
    ret

# ----------------------------------------------------------------------------
# AVX2 micro-kernels differ from AVX-512 in that they have fewer registers,
# and those registers are narrower. We have 16x 256-bit YMM registers.
# ----------------------------------------------------------------------------

tops_f64_avx2_asm_kernel:
    vfmadd231pd %ymm1, %ymm2, %ymm0
    vfmadd231pd %ymm4, %ymm5, %ymm3
    vfmadd231pd %ymm7, %ymm8, %ymm6
    vfmadd231pd %ymm9, %ymm10, %ymm11
    vfmadd231pd %ymm12, %ymm13, %ymm14
    movabsq $40, %rax
    ret

tops_f32_avx2_asm_kernel:
    vfmadd231ps %ymm1, %ymm2, %ymm0
    vfmadd231ps %ymm4, %ymm5, %ymm3
    vfmadd231ps %ymm7, %ymm8, %ymm6
    vfmadd231ps %ymm9, %ymm10, %ymm11
    vfmadd231ps %ymm12, %ymm13, %ymm14
    movabsq $80, %rax
    ret

# ----------------------------------------------------------------------------
# AMX is an Intel-specific extension that introduces a new set of registers
# and instructions for matrix multiplication. Proper usage would require
# loading the tile configuration file, setting up the tile registers, loading
# a rectangular tile of the bigger matrix, performing the multiplications and
# putting back. We will avoid most of those steps!
#
# https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#amxtechs=AMX_BF16,AMX_INT8,AMX_FP16
# ----------------------------------------------------------------------------

tops_bf16_amx_asm_kernel:
    # Each line performs a 16x32x16 matrix multiplication.
    # That's 16x32x16 scalar multiplications and 16x31x16 scalar additions.
    # That's 16,128 scalar operations per line or 64,512 scalar operations total.
    tdpbf16ps %tmm7, %tmm0, %tmm1
    tdpbf16ps %tmm6, %tmm2, %tmm3
    tdpbf16ps %tmm5, %tmm0, %tmm2
    tdpbf16ps %tmm4, %tmm1, %tmm3
    movabsq $64512, %rax
    ret

tops_u8_amx_asm_kernel:
    # Each line performs a 16x64x16 matrix multiplication.
    # That's 16x64x16 scalar multiplications and 16x63x16 scalar additions.
    # That's 32,512 scalar operations per line or 130,048 scalar operations total.
    tdpbuud %tmm7, %tmm0, %tmm1
    tdpbuud %tmm6, %tmm2, %tmm3
    tdpbuud %tmm5, %tmm0, %tmm2
    tdpbuud %tmm4, %tmm1, %tmm3
    movabsq $130048, %rax
    ret

tops_i8_amx_asm_kernel:
    tdpbssd %tmm7, %tmm0, %tmm1
    tdpbssd %tmm6, %tmm2, %tmm3
    tdpbssd %tmm5, %tmm0, %tmm2
    tdpbssd %tmm4, %tmm1, %tmm3
    movabsq $130048, %rax
    ret

# ----------------------------------------------------------------------------