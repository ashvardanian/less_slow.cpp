# ----------------------------------------------------------------------------
# less_slow_amd64.asm
# Micro-kernels for building a performance-first mindset for 64-bit x86.
# ----------------------------------------------------------------------------
# Export symbols so linkers can find them:

    .section .text
    .global i32_add_asm

# ----------------------------------------------------------------------------
# Simple function that adds two 32-bit signed integers using System V AMD64.
# Arguments in 32-bit registers EDI (a) and ESI (b). Return value in EAX.
# ----------------------------------------------------------------------------
i32_add_asm:
    addl    %esi, %edi      # EDI = EDI + ESI
    movl    %edi, %eax      # EAX = EDI
    ret

# ----------------------------------------------------------------------------