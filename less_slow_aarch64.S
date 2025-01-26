# ----------------------------------------------------------------------------
# less_slow_aarch64.asm
# Micro-kernels for building a performance-first mindset for 64-bit ARM (NEON).
# ----------------------------------------------------------------------------

    .section .text
    .global i32_add_asm_kernel
    
    .global tops_f64_neon_asm_kernel
    .global tops_f32_neon_asm_kernel
    .global tops_f16_neon_asm_kernel
    .global tops_bf16_neon_asm_kernel
    .global tops_i8_neon_asm_kernel
    .global tops_u8_neon_asm_kernel

# ----------------------------------------------------------------------------
# Simple function that adds two 32-bit integers. 
# AArch64 ABI: W0 = 'a', W1 = 'b'. Return in W0.
# ----------------------------------------------------------------------------
i32_add_asm_kernel:
    add     w0, w0, w1
    ret

# ----------------------------------------------------------------------------
# f64 micro-kernel:
# Each FMLA vD.2d, vN.2d, vM.2d => 2 multiplies + 2 adds = 4 FLOPs.
# We'll do 10 instructions => 10 × 4 = 40 FLOPs total, returning 40 in W0.
# ----------------------------------------------------------------------------
tops_f64_neon_asm_kernel:
    fmla    v0.2d,  v1.2d,  v2.2d
    fmla    v3.2d,  v4.2d,  v5.2d
    fmla    v6.2d,  v7.2d,  v8.2d
    fmla    v9.2d,  v10.2d, v11.2d
    fmla    v12.2d, v13.2d, v14.2d
    fmla    v15.2d, v16.2d, v17.2d
    fmla    v18.2d, v19.2d, v20.2d
    fmla    v21.2d, v22.2d, v23.2d
    fmla    v24.2d, v25.2d, v26.2d
    fmla    v27.2d, v28.2d, v29.2d

    mov     w0, #40
    ret

# ----------------------------------------------------------------------------
# f32 micro-kernel maximizing FLOPs:
# Each FMLA vD.4s, vN.4s, vM.4s => 4 multiplies + 4 adds = 8 FLOPs.
# Let's do 10 instructions => 10 × 8 = 80 FLOPs total.
# Return 80 in W0.
# ----------------------------------------------------------------------------
tops_f32_neon_asm_kernel:
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
# f16 micro-kernel:
# Requires Armv8.2 half-precision vector arithmetic.
# Each FMLA vD.8h, vN.8h, vM.8h => 8 multiplies + 8 adds = 16 FLOPs.
# We'll do 10 instructions => 160 FLOPs total, returning 160 in W0.
# ----------------------------------------------------------------------------
tops_f16_neon_asm_kernel:
    fmla    v0.8h,  v1.8h,  v2.8h
    fmla    v3.8h,  v4.8h,  v5.8h
    fmla    v6.8h,  v7.8h,  v8.8h
    fmla    v9.8h,  v10.8h, v11.8h
    fmla    v12.8h, v13.8h, v14.8h
    fmla    v15.8h, v16.8h, v17.8h
    fmla    v18.8h, v19.8h, v20.8h
    fmla    v21.8h, v22.8h, v23.8h
    fmla    v24.8h, v25.8h, v26.8h
    fmla    v27.8h, v28.8h, v29.8h

    mov     w0, #160
    ret

# ----------------------------------------------------------------------------
# bf16 micro-kernel:
# Requires Armv8.6 BF16 instructions (BFMMLA, etc.).
# bfmmla  vD.4s, vN.8h, vM.8h => 8 multiplies + 8 adds = 16 FLOPs.
# We'll do 10 instructions => 160 FLOPs total, returning 160 in W0.
# ----------------------------------------------------------------------------
tops_bf16_neon_asm_kernel:
    bfmmla  v0.4s,  v1.8h,  v2.8h
    bfmmla  v3.4s,  v4.8h,  v5.8h
    bfmmla  v6.4s,  v7.8h,  v8.8h
    bfmmla  v9.4s,  v10.8h, v11.8h
    bfmmla  v12.4s, v13.8h, v14.8h
    bfmmla  v15.4s, v16.8h, v17.8h
    bfmmla  v18.4s, v19.8h, v20.8h
    bfmmla  v21.4s, v22.8h, v23.8h
    bfmmla  v24.4s, v25.8h, v26.8h
    bfmmla  v27.4s, v28.8h, v29.8h    

    mov     w0, #160
    ret

# ----------------------------------------------------------------------------
# i8 micro-kernel:
# Requires Armv8.4 sdot or i8mm extension.
# sdot vD.4s, vN.16b, vM.16b => 16 multiplies + 16 adds = 32 FLOPs.
# We'll do 10 instructions => 320 FLOPs total, returning 320 in W0.
# ----------------------------------------------------------------------------
tops_i8_neon_asm_kernel:
    sdot    v0.4s,  v1.16b,  v2.16b
    sdot    v3.4s,  v4.16b,  v5.16b
    sdot    v6.4s,  v7.16b,  v8.16b
    sdot    v9.4s,  v10.16b, v11.16b
    sdot    v12.4s, v13.16b, v14.16b
    sdot    v15.4s, v16.16b, v17.16b
    sdot    v18.4s, v19.16b, v20.16b
    sdot    v21.4s, v22.16b, v23.16b
    sdot    v24.4s, v25.16b, v26.16b
    sdot    v27.4s, v28.16b, v29.16b

    mov     w0, #320
    ret

# ----------------------------------------------------------------------------
# u8 micro-kernel:
# Requires Armv8.4 i8mm extension.
# udot vD.4s, vN.16b, vM.16b => 16 multiplies + 16 adds = 32 FLOPs.
# We'll do 10 instructions => 320 FLOPs total, returning 320 in W0.
# ----------------------------------------------------------------------------
tops_u8_neon_asm_kernel:
    udot    v0.4s,  v1.16b,  v2.16b
    udot    v3.4s,  v4.16b,  v5.16b
    udot    v6.4s,  v7.16b,  v8.16b
    udot    v9.4s,  v10.16b, v11.16b
    udot    v12.4s, v13.16b, v14.16b
    udot    v15.4s, v16.16b, v17.16b
    udot    v18.4s, v19.16b, v20.16b
    udot    v21.4s, v22.16b, v23.16b
    udot    v24.4s, v25.16b, v26.16b
    udot    v27.4s, v28.16b, v29.16b

    mov     w0, #320
    ret

# ----------------------------------------------------------------------------
# Tell the linker/assembler that we do NOT need an executable stack:
    .section .note.GNU-stack, "", @progbits
# ----------------------------------------------------------------------------