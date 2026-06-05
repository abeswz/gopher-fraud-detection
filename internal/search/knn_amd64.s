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

// func computeClusterBatch8(minSoA, maxSoA *int16, q *[16]int16, lbs *[8]int32)
//
// Computes bbox lower bounds for 8 clusters using bpsoaMin/bpsoaMax layout.
// Each group has NPairs*16 = 112 int16s; pair p is at byte offset p*32.
// For pair p, query dims are q[2p] and q[2p+1] (broadcast as i32 to all 8 cluster lanes).
// gap = max(lo-q,0) + max(q-hi,0); lb[c] = sum_p( gap[c,2p]² + gap[c,2p+1]² ).
// Result: 8 int32 in lbs; may overflow (go negative) for distant clusters — caller clamps.
//
// Argument layout (ABI0, stack-based):
//   minSoA  = 0(FP)
//   maxSoA  = 8(FP)
//   q       = 16(FP)
//   lbs     = 24(FP)
TEXT ·computeClusterBatch8(SB), NOSPLIT, $0-32
    MOVQ minSoA+0(FP), SI
    MOVQ maxSoA+8(FP), DI
    MOVQ q+16(FP), DX
    MOVQ lbs+24(FP), R8

    VPXOR Y7, Y7, Y7    // Y7 = accumulator (8 × i32 = 0)
    VPXOR Y6, Y6, Y6    // Y6 = zero for VPMAXSW

    // Pair 0: dims 0,1 — minSoA/maxSoA offset 0, q byte offset 0
    VPBROADCASTD 0(DX), Y5
    VMOVDQU 0(SI), Y0
    VMOVDQU 0(DI), Y1
    VPSUBW Y5, Y0, Y2       // Y2 = lo - q
    VPMAXSW Y6, Y2, Y2      // Y2 = max(lo-q, 0)
    VPSUBW Y1, Y5, Y3       // Y3 = q - hi
    VPMAXSW Y6, Y3, Y3      // Y3 = max(q-hi, 0)
    VPADDW Y2, Y3, Y4       // Y4 = gap (one of the two terms is 0 for valid bbox)
    VPMADDWD Y4, Y4, Y4     // Y4 = 8×i32: gap[2i]²+gap[2i+1]²
    VPADDD Y7, Y4, Y7

    // Pair 1: dims 2,3 — offset 32, q byte 4
    VPBROADCASTD 4(DX), Y5
    VMOVDQU 32(SI), Y0
    VMOVDQU 32(DI), Y1
    VPSUBW Y5, Y0, Y2
    VPMAXSW Y6, Y2, Y2
    VPSUBW Y1, Y5, Y3
    VPMAXSW Y6, Y3, Y3
    VPADDW Y2, Y3, Y4
    VPMADDWD Y4, Y4, Y4
    VPADDD Y7, Y4, Y7

    // Pair 2: dims 4,5 — offset 64, q byte 8
    VPBROADCASTD 8(DX), Y5
    VMOVDQU 64(SI), Y0
    VMOVDQU 64(DI), Y1
    VPSUBW Y5, Y0, Y2
    VPMAXSW Y6, Y2, Y2
    VPSUBW Y1, Y5, Y3
    VPMAXSW Y6, Y3, Y3
    VPADDW Y2, Y3, Y4
    VPMADDWD Y4, Y4, Y4
    VPADDD Y7, Y4, Y7

    // Pair 3: dims 6,7 — offset 96, q byte 12
    VPBROADCASTD 12(DX), Y5
    VMOVDQU 96(SI), Y0
    VMOVDQU 96(DI), Y1
    VPSUBW Y5, Y0, Y2
    VPMAXSW Y6, Y2, Y2
    VPSUBW Y1, Y5, Y3
    VPMAXSW Y6, Y3, Y3
    VPADDW Y2, Y3, Y4
    VPMADDWD Y4, Y4, Y4
    VPADDD Y7, Y4, Y7

    // Pair 4: dims 8,9 — offset 128, q byte 16
    VPBROADCASTD 16(DX), Y5
    VMOVDQU 128(SI), Y0
    VMOVDQU 128(DI), Y1
    VPSUBW Y5, Y0, Y2
    VPMAXSW Y6, Y2, Y2
    VPSUBW Y1, Y5, Y3
    VPMAXSW Y6, Y3, Y3
    VPADDW Y2, Y3, Y4
    VPMADDWD Y4, Y4, Y4
    VPADDD Y7, Y4, Y7

    // Pair 5: dims 10,11 — offset 160, q byte 20
    VPBROADCASTD 20(DX), Y5
    VMOVDQU 160(SI), Y0
    VMOVDQU 160(DI), Y1
    VPSUBW Y5, Y0, Y2
    VPMAXSW Y6, Y2, Y2
    VPSUBW Y1, Y5, Y3
    VPMAXSW Y6, Y3, Y3
    VPADDW Y2, Y3, Y4
    VPMADDWD Y4, Y4, Y4
    VPADDD Y7, Y4, Y7

    // Pair 6: dims 12,13 — offset 192, q byte 24
    VPBROADCASTD 24(DX), Y5
    VMOVDQU 192(SI), Y0
    VMOVDQU 192(DI), Y1
    VPSUBW Y5, Y0, Y2
    VPMAXSW Y6, Y2, Y2
    VPSUBW Y1, Y5, Y3
    VPMAXSW Y6, Y3, Y3
    VPADDW Y2, Y3, Y4
    VPMADDWD Y4, Y4, Y4
    VPADDD Y7, Y4, Y7

    VMOVDQU Y7, 0(R8)
    VZEROALL
    RET
