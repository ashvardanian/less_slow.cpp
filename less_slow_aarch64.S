# less_slow_aarch64.S
# Micro-kernels for building a performance-first mindset for 64-bit ARM (NEON).

#ifdef __APPLE__
    #define SYMBOL_NAME(name) _##name  // Add underscore on macOS
#else
    #define SYMBOL_NAME(name) name     // No underscore on GNU-based systems
#endif

    .text
    .global SYMBOL_NAME(i32_add_asm_kernel)

    .global SYMBOL_NAME(tops_f64_neon_asm_kernel)
    .global SYMBOL_NAME(tops_f32_neon_asm_kernel)
    .global SYMBOL_NAME(tops_f16_neon_asm_kernel)
    .global SYMBOL_NAME(tops_bf16_neon_asm_kernel)
    .global SYMBOL_NAME(tops_i8_neon_asm_kernel)
    .global SYMBOL_NAME(tops_u8_neon_asm_kernel)

# Simple function that adds two 32-bit integers.
# AArch64 ABI: W0 = 'a', W1 = 'b'. Return in W0.
SYMBOL_NAME(i32_add_asm_kernel):
    add     w0, w0, w1
    ret

# f64 micro-kernel:
# Each FMLA vD.2d, vN.2d, vM.2d => 2 multiplies + 2 adds = 4 FLOPs.
# We'll do 10 instructions => 10 × 4 = 40 FLOPs total, returning 40 in W0.
#
# Every kernel below accumulates into V0-V7, V16, V17 and reads V18-V31.
# AAPCS64 makes the lower halves of V8-V15 callee-saved, so touching them
# would cost us a prologue and epilogue. That leaves 24 usable registers
# against the 30 ten distinct FMLA triples would need, so - like a real GEMM
# micro-kernel - we share the read-only multiplicands and keep only the 10
# accumulators unique. The 10 dependency chains stay independent either way.
SYMBOL_NAME(tops_f64_neon_asm_kernel):
    fmla    v0.2d,  v18.2d, v19.2d
    fmla    v1.2d,  v20.2d, v21.2d
    fmla    v2.2d,  v22.2d, v23.2d
    fmla    v3.2d,  v24.2d, v25.2d
    fmla    v4.2d,  v26.2d, v27.2d
    fmla    v5.2d,  v28.2d, v29.2d
    fmla    v6.2d,  v30.2d, v31.2d
    fmla    v7.2d,  v18.2d, v20.2d
    fmla    v16.2d, v22.2d, v24.2d
    fmla    v17.2d, v26.2d, v28.2d

    mov     w0, #40
    ret

# f32 micro-kernel maximizing FLOPs:
# Each FMLA vD.4s, vN.4s, vM.4s => 4 multiplies + 4 adds = 8 FLOPs.
# Let's do 10 instructions => 10 × 8 = 80 FLOPs total.
# Return 80 in W0.
SYMBOL_NAME(tops_f32_neon_asm_kernel):
    fmla    v0.4s,  v18.4s, v19.4s
    fmla    v1.4s,  v20.4s, v21.4s
    fmla    v2.4s,  v22.4s, v23.4s
    fmla    v3.4s,  v24.4s, v25.4s
    fmla    v4.4s,  v26.4s, v27.4s
    fmla    v5.4s,  v28.4s, v29.4s
    fmla    v6.4s,  v30.4s, v31.4s
    fmla    v7.4s,  v18.4s, v20.4s
    fmla    v16.4s, v22.4s, v24.4s
    fmla    v17.4s, v26.4s, v28.4s

    mov     w0, #80
    ret

# f16 micro-kernel:
# Requires Armv8.2 half-precision vector arithmetic.
# Each FMLA vD.8h, vN.8h, vM.8h => 8 multiplies + 8 adds = 16 FLOPs.
# We'll do 10 instructions => 160 FLOPs total, returning 160 in W0.
SYMBOL_NAME(tops_f16_neon_asm_kernel):
    fmla    v0.8h,  v18.8h, v19.8h
    fmla    v1.8h,  v20.8h, v21.8h
    fmla    v2.8h,  v22.8h, v23.8h
    fmla    v3.8h,  v24.8h, v25.8h
    fmla    v4.8h,  v26.8h, v27.8h
    fmla    v5.8h,  v28.8h, v29.8h
    fmla    v6.8h,  v30.8h, v31.8h
    fmla    v7.8h,  v18.8h, v20.8h
    fmla    v16.8h, v22.8h, v24.8h
    fmla    v17.8h, v26.8h, v28.8h

    mov     w0, #160
    ret

# bf16 micro-kernel:
# Requires Armv8.6 BF16 instructions (BFMMLA, etc.).
# bfmmla  vD.4s, vN.8h, vM.8h => 8 multiplies + 8 adds = 16 FLOPs.
# We'll do 10 instructions => 160 FLOPs total, returning 160 in W0.
SYMBOL_NAME(tops_bf16_neon_asm_kernel):
    bfmmla  v0.4s,  v18.8h, v19.8h
    bfmmla  v1.4s,  v20.8h, v21.8h
    bfmmla  v2.4s,  v22.8h, v23.8h
    bfmmla  v3.4s,  v24.8h, v25.8h
    bfmmla  v4.4s,  v26.8h, v27.8h
    bfmmla  v5.4s,  v28.8h, v29.8h
    bfmmla  v6.4s,  v30.8h, v31.8h
    bfmmla  v7.4s,  v18.8h, v20.8h
    bfmmla  v16.4s, v22.8h, v24.8h
    bfmmla  v17.4s, v26.8h, v28.8h

    mov     w0, #160
    ret

# i8 micro-kernel:
# Requires Armv8.4 sdot or i8mm extension.
# sdot vD.4s, vN.16b, vM.16b => 16 multiplies + 16 adds = 32 FLOPs.
# We'll do 10 instructions => 320 FLOPs total, returning 320 in W0.
SYMBOL_NAME(tops_i8_neon_asm_kernel):
    sdot    v0.4s,  v18.16b, v19.16b
    sdot    v1.4s,  v20.16b, v21.16b
    sdot    v2.4s,  v22.16b, v23.16b
    sdot    v3.4s,  v24.16b, v25.16b
    sdot    v4.4s,  v26.16b, v27.16b
    sdot    v5.4s,  v28.16b, v29.16b
    sdot    v6.4s,  v30.16b, v31.16b
    sdot    v7.4s,  v18.16b, v20.16b
    sdot    v16.4s, v22.16b, v24.16b
    sdot    v17.4s, v26.16b, v28.16b

    mov     w0, #320
    ret

# u8 micro-kernel:
# Requires Armv8.4 i8mm extension.
# udot vD.4s, vN.16b, vM.16b => 16 multiplies + 16 adds = 32 FLOPs.
# We'll do 10 instructions => 320 FLOPs total, returning 320 in W0.
SYMBOL_NAME(tops_u8_neon_asm_kernel):
    udot    v0.4s,  v18.16b, v19.16b
    udot    v1.4s,  v20.16b, v21.16b
    udot    v2.4s,  v22.16b, v23.16b
    udot    v3.4s,  v24.16b, v25.16b
    udot    v4.4s,  v26.16b, v27.16b
    udot    v5.4s,  v28.16b, v29.16b
    udot    v6.4s,  v30.16b, v31.16b
    udot    v7.4s,  v18.16b, v20.16b
    udot    v16.4s, v22.16b, v24.16b
    udot    v17.4s, v26.16b, v28.16b

    mov     w0, #320
    ret

# Tell the linker/assembler that we do NOT need an executable stack:
#ifdef __linux__
    .section .note.GNU-stack, "", @progbits
#endif
