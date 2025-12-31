#pragma once

#include "fast_prng.h"
#include <stdint.h>

/** @file tsetlin_machine.h
 *  @brief Public API for the (dense) Tsetlin Machine implementation.
 */

/** @brief Tsetlin Machine (dense) internal state.
 *
 *  Don't create, modify or free this struct directly; use the provided
 *  constructors and destructors (tm_create, tm_free, etc.).
 */
struct TsetlinMachine {
	uint32_t num_classes;
	uint32_t threshold;
	uint32_t num_literals;
	uint32_t num_clauses;
	int8_t max_state, min_state;
	uint8_t boost_true_positive_feedback;
	float s;

	uint32_t y_size, y_element_size;
	uint8_t (*y_eq)(const struct TsetlinMachine *tm, const void *y, const void *y_pred);
	void (*output_activation)(const struct TsetlinMachine *tm, const void *y_pred);
	void (*calculate_feedback)(struct TsetlinMachine *tm, const uint8_t *X,
							   const void *y);

	int8_t mid_state;
	float s_inv, s_min1_inv;
	int8_t *ta_state;		// shape: flat (num_clauses, num_literals, 2)
	int16_t *weights;		// shape: flat (num_clauses, num_classes)
	uint8_t *clause_output; // shape: (num_clauses)
	int32_t *votes;			// shape: (num_classes)

	struct FastPRNG rng;
};

/**
 *  @note Data layout conventions used across the API:
 *  - X: flat array of shape (rows, num_literals) of uint8_t (each value 0 or 1)
 *  - y: flat array of shape (rows, y_size) with element size y_element_size
 *  - y_pred: flat array of shape (rows, num_classes) with element size y_element_size
 */

/**
 * @brief Create and initialize a Tsetlin Machine instance.
 *
 * @param num_classes Number of classes to predict.
 * @param threshold Threshold for clause votes.
 * @param num_literals Number of literals (features) in the input data.
 * @param num_clauses Number of clauses in the Tsetlin Machine.
 * @param max_state Maximum automaton state value.
 * @param min_state Minimum automaton state value.
 * @param boost_true_positive_feedback Whether to boost true positive feedback (0/1).
 * @param y_size Size of the output vector (e.g., 1 for single-label classification).
 * @param y_element_size Size of each output element (in bytes).
 * @param s Sensitivity / learning parameter (> 1.0).
 * @param seed Random seed for reproducibility.
 * @return Pointer to the created TsetlinMachine, or NULL on allocation failure.
 */
struct TsetlinMachine *tm_create(uint32_t num_classes, uint32_t threshold,
								 uint32_t num_literals, uint32_t num_clauses,
								 int8_t max_state, int8_t min_state,
								 uint8_t boost_true_positive_feedback, uint32_t y_size,
								 uint32_t y_element_size, float s, uint32_t seed);

// Load Tsetlin Machine from a bin file
struct TsetlinMachine *tm_load(const char *filename, uint32_t y_size,
							   uint32_t y_element_size);

// Save Tsetlin Machine to a bin file
void tm_save(const struct TsetlinMachine *tm, const char *filename);

#if BUILD_FLATCC
// Load Tsetlin Machine from a flatbuffers file
struct TsetlinMachine *tm_load_fbs(const char *filename, uint32_t y_size,
								   uint32_t y_element_size);

// Save Tsetlin Machine to a flatbuffers file
void tm_save_fbs(struct TsetlinMachine *tm, const char *filename);
#endif

// Free all allocated memory
// It also frees the TsetlinMachine struct itself
// Remember to set tm to NULL after this call
void tm_free(struct TsetlinMachine *tm);

// Train
// X shape: flat (rows, num_literals) of uint8_t (each uint8_t should be 0 or 1)
// y shape: flat (rows, y_size) with element size (y_element_size) of any type (void *)
void tm_train(struct TsetlinMachine *tm, const uint8_t *X, const void *y, uint32_t rows,
			  uint32_t epochs);

// Inference
// Writes to user allocated memory y_pred
// X shape: flat (rows, num_literals) of uint8_t (each uint8_t should be 0 or 1)
// y_pred shape: flat (rows, num_classes) with element size (y_element_size) of any type
// (void *)
void tm_predict(struct TsetlinMachine *tm, const uint8_t *X, void *y_pred, uint32_t rows);

// Simple accuracy evaluation
// X shape: flat (rows, num_literals) of uint8_t (each uint8_t should be 0 or 1)
// y shape: flat (rows, y_size) with element size (y_element_size) of any type (void *)
void tm_evaluate(struct TsetlinMachine *tm, const uint8_t *X, const void *y,
				 uint32_t rows);

// --- y_eq ---
// Since y and y_pred are of any type, this function determines whether y == y_pred
// You can write your own y_eq function, and set it in the TsetlinMachine struct

// Basic y_eq function comparing raw memory using memcmp
// Works with any trivial types
uint8_t tm_y_eq_generic(const struct TsetlinMachine *tm, const void *y,
						const void *y_pred);

// --- \/ DON'T USE THESE FUNCTIONS DIRECTLY \/ ---
// Unless you know what you are doing, use the tm_set_* functions to set the desired
// components or leave them as default

// --- output_activation ---
// The raw output of a Tsetlin Machine are just summed up votes (tm->votes), of shape
// (num_classes) This function translates votes into a desirable format of any type (void
// *)

// Output is a class index (e.g., for classification tasks)
// y_size = 1, y_element_size = sizeof(uint32_t)
void tm_oa_class_idx(const struct TsetlinMachine *tm, const void *y_pred);

// Output is a binary vector of class predictions
// y_size = tm->num_classes, y_element_size = sizeof(uint8_t)
void tm_oa_bin_vector(const struct TsetlinMachine *tm, const void *y_pred);

// Set the output activation function
// Provided functions are tm_oa_class_idx and tm_oa_bin_vector
// Or implement your own
// Default is tm_oa_class_idx
void tm_set_output_activation(struct TsetlinMachine *tm,
							  void (*output_activation)(const struct TsetlinMachine *tm,
														const void *y_pred));

// --- calculate_feedback ---
// Based on the raw(!) output of a Tsetlin Machine (tm->votes) and ground truth labels
// (y), decide which clause-class pairs to update

// Ground truth label is a class index (e.g., for classification tasks)
// y_size = 1, y_element_size = sizeof(uint32_t)
void tm_feedback_class_idx(struct TsetlinMachine *tm, const uint8_t *X, const void *y);

// Ground truth label is a binary vector of class predictions
// y_size = tm->num_classes, y_element_size = sizeof(uint8_t)
void tm_feedback_bin_vector(struct TsetlinMachine *tm, const uint8_t *X, const void *y);

// Internal component of feedback functions, included in header if you want to create your
// own Decide which clause-class pairs to update
void tm_apply_feedback(struct TsetlinMachine *tm, uint32_t clause_id, uint32_t class_id,
					   uint8_t is_class_positive, const uint8_t *X);

// Set the feedback function
// Provided functions are tm_feedback_class_idx and tm_feedback_bin_vector
// Or implement your own
// Default is tm_feedback_class_idx
void tm_set_calculate_feedback(struct TsetlinMachine *tm,
							   void (*calculate_feedback)(struct TsetlinMachine *tm,
														  const uint8_t *X,
														  const void *y));
