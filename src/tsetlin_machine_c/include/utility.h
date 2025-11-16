/** @file utility.h
 *  @brief Small utility helpers used across the Tsetlin Machine implementation.
 */

#pragma once

#include <stdint.h>

/**
 * @brief Clip an integer to +/- threshold.
 * @param x Value to clip.
 * @param threshold Positive threshold value.
 * @return Clipped value in [-threshold, threshold].
 */
static inline int32_t clip(const int32_t x, const int32_t threshold) {
    if (x > threshold) return threshold;
    else if (x < -threshold) return -threshold;
    return x;
}

/**
 * @brief Type-generic minimum macro.
 * @note This is a GCC/Clang statement-expression; behaviour on other compilers
 *       may be undefined. Keep usage minimal to avoid portability issues.
 */
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

