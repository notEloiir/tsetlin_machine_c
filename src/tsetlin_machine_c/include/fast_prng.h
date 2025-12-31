#pragma once

#include <stdint.h>

/** @file fast_prng.h
 *  @brief Small, fast pseudorandom number generator based on xorshift32.
 */

/** @brief PRNG state for xorshift32.
 */
struct FastPRNG {
	uint32_t state;
};

/**
 * @brief Seed the PRNG.
 * @param prng PRNG instance to seed.
 * @param seed Seed value (0 is replaced with a fixed non-zero default).
 */
void prng_seed(struct FastPRNG *prng, uint32_t seed);

/**
 * @brief Generate a pseudo-random 32-bit unsigned integer in [0, UINT32_MAX].
 * @param prng PRNG instance.
 * @return Random uint32_t.
 */
uint32_t prng_next_uint32(struct FastPRNG *prng);

/**
 * @brief Generate a pseudo-random float in [0, 1).
 * @param prng PRNG instance.
 * @return Random float in [0,1).
 */
float prng_next_float(struct FastPRNG *prng);
