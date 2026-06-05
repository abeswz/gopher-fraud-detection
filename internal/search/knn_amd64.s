// Copyright: see project license.
#include "textflag.h"

// func distL2i16q(vecs []int16, base int, q *[16]int16) int32
//
// Computes L2² in int16 space: sum of (vecs[base+i] - q[i])² for i=0..15.
// No float conversion. VPSUBW + VPMADDWD (2 dims per lane per iter).
// Safe range: max total = 2×(20000)²+12×(10000)² = 2e9 < int32_max.
//
// Argument layout (ABI0, stack-based):
//   vecs.ptr  = 0(FP)
//   vecs.len  = 8(FP)
//   vecs.cap  = 16(FP)
//   base      = 24(FP)
//   q         = 32(FP)
//   ret       = 40(FP), 4 bytes
TEXT ·distL2i16q(SB), NOSPLIT, $0-44
    MOVQ vecs+0(FP), SI
    MOVQ base+24(FP), AX
    MOVQ q+32(FP), DI

    // SI → &vecs[base]
    LEAQ (SI)(AX*2), SI

    // Load 16×int16 from vecs and query (32 bytes each)
    VMOVDQU (SI), Y0          // Y0 = vecs[base..base+15]
    VMOVDQU (DI), Y1          // Y1 = q[0..15]

    // diff = vecs - query (i16, no overflow: |diff| ≤ 20000 < 32767)
    VPSUBW Y1, Y0, Y0         // Y0 = Y0 - Y1

    // Square pairs: diff[2i]²+diff[2i+1]² → i32 per lane (8 lanes)
    VPMADDWD Y0, Y0, Y0       // Y0 = 8×i32 partial sums

    // Reduce 8 i32 → 1 i32
    VEXTRACTI128 $1, Y0, X1   // X1 = upper 4 i32
    VPADDD X1, X0, X0         // X0 = lower + upper (4 i32)
    VPHADDD X0, X0, X0        // X0 = [s0+s1, s2+s3, ...]
    VPHADDD X0, X0, X0        // X0[0] = total sum

    VMOVD X0, ret+40(FP)
    VZEROALL
    RET
