# GEMM_U8_P32X4_V1

Canonical input is a fixed-width byte vector of K bytes, 1 <= K <= 33025.
Different widths are different hash domains. p = 4294967291 is prime.

The SHAKE256 input is, in this exact order:

1. ASCII `MGBFS/GEMM_U8_P32X4/V1` followed by one zero byte.
2. K as uint32 little endian.
3. The 16 seed bytes in declared order (numeric seeds are encoded little endian).

Read uint32 little-endian words from the XOF. Reject words >= p. Fill
`a[position][lane]` in position-major, lane-minor order, then four offsets b.

For lane j, h_j(x) = (b_j + sum_i x_i*a_ij) mod p. Hash bytes are four uint32
residues in lane order, each little endian. Numeric sorting treats lane 3 as the
most significant word. Ownership consumes that same numeric high-bit prefix.

## Tensor Core layout

X: candidate-major uint8 [candidate_count, padded_K], zero padded to a supported
K tile. R: uint8 [padded_K,16]. R[i,4*j+l] is byte l of a_ij; padding rows are zero.
Unsigned-byte MMA with signed-int32 accumulators computes S = X R exactly.
Each S element is at most K*255*255 <= INT32_MAX. Epilogue reconstructs
`sum_l uint64(S[4*j+l]) << (8*l)`, adds b_j and reduces modulo p. That sum fits
uint64 for every supported K. Saturating MMA must not hide invalid input bounds.

## Guarantees and limits

For independent uniform coefficients over F_p, two distinct fixed byte vectors
collide with probability exactly p^-4. Union bound for N fixed inputs is
N*(N-1)/(2*p^4). A SHAKE-expanded 128-bit seed is not literally an independent
uniform draw of the entire coefficient matrix: using that bound for seeded
production is a pseudorandom-XOF assumption, not an information-theoretic proof.
The family is linear and is not intended for adversarial collision resistance.
Changing seed gives a new verification run, not an exactness certificate.

## Independent frozen vectors

K=16, all-zero seed. Python hashlib SHAKE256, integer modular dot products:

- zero bytes: [1710827310,2245209978,2416263789,1202685372]
- bytes 0..15: [2244859959,2401010834,3228855414,2263550226]

CPU scalar, limb reconstruction and GPU implementations must match these values.
