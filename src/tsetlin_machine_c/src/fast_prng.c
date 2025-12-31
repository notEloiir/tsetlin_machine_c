#include "fast_prng.h"

/**
 * @brief Seed the xorshift32 PRNG.
 * @param prng PRNG instance to seed.
 * @param seed Seed value (0 replaced with default non-zero constant).
 */
void prng_seed(struct FastPRNG *prng, uint32_t seed) {
	/* Seed should not be zero for xorshift32 */
	prng->state = seed ? seed : 0xdeadbeef;
}

/**
 * @brief Generate the next 32-bit pseudorandom integer.
 * @param prng PRNG instance.
 * @return Pseudorandom uint32_t.
 */
uint32_t prng_next_uint32(struct FastPRNG *prng) {
	uint32_t x = prng->state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	prng->state = x;
	return x;
}

/**
 * @brief Generate a pseudorandom float in [0,1).
 *
 * Uses bit-manipulation to construct a float from random mantissa bits.
 * @param prng PRNG instance.
 * @return Pseudorandom float in [0,1).
 */
float prng_next_float(struct FastPRNG *prng) {
	uint32_t x = prng->state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	prng->state = x;

	static union {
		uint32_t u32;
		float f;
	} caster;

	/* Reinterpret x as float: keep random mantissa, set exponent to 127
	   (1.0f), then subtract 1.0f to obtain value in [0,1). */
	caster.u32 = 0x3F800000 | x >> 9;

	return caster.f - 1.0f;
}
