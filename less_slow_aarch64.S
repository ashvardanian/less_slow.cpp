# ----------------------------------------------------------------------------
# less_slow_aarch64.asm
# Micro-kernels for building a performance-first mindset for 64-bit ARM (NEON).
# ----------------------------------------------------------------------------

    .section .text
    .global i32_add_asm_kernel
    .global f32_matmul_neon_flops_asm_kernel

# ----------------------------------------------------------------------------
# Simple function that adds two 32-bit signed integers using AArch64 ABI.
# Arguments in W0 (a) and W1 (b). Return value in W0.
# ----------------------------------------------------------------------------
i32_add_asm_kernel:
    add     w0, w0, w1
    ret

# ----------------------------------------------------------------------------
# NEON micro-kernel to maximize FLOPs.
# Each FMLA vD.4s, vN.4s, vM.4s => 4 multiplies + 4 adds = 8 FLOPs.
# We'll do 10 instructions => 80 FLOPs total.
# ----------------------------------------------------------------------------
f32_matmul_neon_flops_asm_kernel:
    fmla    v0.4s,  v1.4s,  v2.4s
    fmla    v3.4s,  v4.4s,  v5.4s
    fmla    v6.4s,  v7.4s,  v8.4s
    fmla    v9.4s,  v10.4s, v11.4s
    fmla    v12.4s, v13.4s, v14.4s
    fmla    v15.4s, v16.4s, v17.4s
    fmla    v18.4s, v19.4s, v20.4s
    fmla    v21.4s, v22.4s, v23.4s
    fmla    v24.4s, v25.4s, v26.4s
    fmla    v27.4s, v28.4s, v29.4s

    mov     w0, #80
    ret

# ----------------------------------------------------------------------------
