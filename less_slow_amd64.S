# ----------------------------------------------------------------------------
# less_slow_amd64.asm
# Micro-kernels for building a performance-first mindset for 64-bit x86.
# ----------------------------------------------------------------------------
# Export symbols so linkers can find them:

    .section .text
    .global i32_add_asm_kernel
    .global tops_f32_avx512_asm_kernel

# ----------------------------------------------------------------------------
# Simple function that adds two 32-bit signed integers using System V AMD64.
# Arguments in 32-bit registers EDI (a) and ESI (b). Return value in EAX.
# ----------------------------------------------------------------------------
i32_add_asm_kernel:
    addl    %esi, %edi      # EDI = EDI + ESI
    movl    %edi, %eax      # EAX = EDI
    ret

# ----------------------------------------------------------------------------
# AVX-512 micro-kernel maximizing FLOPs across all ZMM registers.
# ----------------------------------------------------------------------------

tops_f32_avx512_asm_kernel:
    #
    # Each vfmadd231ps does: DEST = DEST + (SRC1 * SRC2)
    # That is 16 multiplies + 16 adds = 32 FLOPs per instruction.
    # We'll do 10 instructions below => 10 × 32 = 320 FLOPs total.
    #
    vfmadd231ps %zmm1,  %zmm2,  %zmm0    # 1
    vfmadd231ps %zmm4,  %zmm5,  %zmm3    # 2
    vfmadd231ps %zmm7,  %zmm8,  %zmm6    # 3
    vfmadd231ps %zmm10, %zmm11, %zmm9    # 4
    vfmadd231ps %zmm13, %zmm14, %zmm12   # 5
    vfmadd231ps %zmm16, %zmm17, %zmm15   # 6
    vfmadd231ps %zmm19, %zmm20, %zmm18   # 7
    vfmadd231ps %zmm22, %zmm23, %zmm21   # 8
    vfmadd231ps %zmm25, %zmm26, %zmm24   # 9
    vfmadd231ps %zmm28, %zmm29, %zmm27   # 10

    # Return value in 64-bit register %rax => 320
    movabsq $320, %rax
    ret

# ----------------------------------------------------------------------------