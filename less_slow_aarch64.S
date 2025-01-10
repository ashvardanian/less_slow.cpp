# ----------------------------------------------------------------------------
# less_slow_aarch64.asm
# Micro-kernels for building a performance-first mindset for 64-bit ARM.
# ----------------------------------------------------------------------------
# Export symbols so linkers can find them:

    .section .text
    .global i32_add_asm

# ----------------------------------------------------------------------------
# Simple function that adds two 32-bit signed integers using AArch64 ABI.
# Arguments in W0 (a) and W1 (b). Return value in W0.
# ----------------------------------------------------------------------------
i32_add_asm:
    add     w0, w0, w1      # w0 = w0 + w1
    ret

# ----------------------------------------------------------------------------