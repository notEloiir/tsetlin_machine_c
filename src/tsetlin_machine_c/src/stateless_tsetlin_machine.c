#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stateless_tsetlin_machine.h"
#include "utility.h"

/**
 * @brief Insert a TANode into a linked list.
 *
 * If prev is NULL the node is inserted at the head. On allocation failure
 * the function prints an error and exits the program.
 */
void ta_stateless_insert(struct TANode **head_ptr, struct TANode *prev, uint32_t ta_id,
						 struct TANode **result) {
	struct TANode *node = malloc(sizeof(struct TANode));
	if (node == NULL) {
		perror("Memory allocation failed");
		exit(1);
	}
	node->ta_id = ta_id;
	node->next = prev != NULL ? prev->next : NULL;

	if (*head_ptr == NULL) {
		*head_ptr = node;
	} else if (prev == NULL) {
		node->next = *head_ptr;
		*head_ptr = node;
	} else {
		prev->next = node;
	}

	if (result != NULL) {
		*result = node;
	}
}

/**
 * @brief Remove a TANode from a linked list.
 *
 * If prev is NULL the head is removed. If the list is empty the function
 * prints a diagnostic and returns.
 */
void ta_stateless_remove(struct TANode **head_ptr, struct TANode *prev,
						 struct TANode **result) {
	if (*head_ptr == NULL) {
		fprintf(stderr, "Trying to remove from empty linked list\n");
		return;
	}
	if (prev != NULL && prev->next == NULL) {
		// Trying to remove node after tail of linked list
		return;
	}

	struct TANode *to_remove = NULL;
	if (prev == NULL) {
		// Removing first element
		to_remove = *head_ptr;
		*head_ptr = to_remove->next;
	} else {
		to_remove = prev->next;
		prev->next = to_remove->next;
	}

	if (result != NULL) {
		*result = to_remove->next;
	}

	free(to_remove);
}

/**
 * @brief Generic comparator for stateless machine outputs.
 *
 * Compares raw bytes of y and y_pred using the configured element size.
 */
uint8_t sltm_y_eq_generic(const struct StatelessTsetlinMachine *sltm, const void *y,
						  const void *y_pred) {
	return 0 == memcmp(y, y_pred, sltm->y_size * sltm->y_element_size);
}

// --- Tsetlin Machine ---

/**
 * @brief Initialize a stateless machine's internal derived fields.
 * @param sltm Pointer to the stateless machine.
 */
void sltm_initialize(struct StatelessTsetlinMachine *sltm);

/**
 * @brief Free all clause linked-lists in a stateless machine.
 */
inline static void sltm_free_state_llists(struct StatelessTsetlinMachine *sltm);

/**
 * @brief Allocate and initialize a StatelessTsetlinMachine instance.
 *
 * @return Pointer to the created instance or NULL on allocation failure.
 */
struct StatelessTsetlinMachine *sltm_create(uint32_t num_classes, uint32_t threshold,
											uint32_t num_literals, uint32_t num_clauses,
											int8_t max_state, int8_t min_state,
											uint8_t boost_true_positive_feedback,
											uint32_t y_size, uint32_t y_element_size,
											float s) {
	struct StatelessTsetlinMachine *sltm =
		(struct StatelessTsetlinMachine *)malloc(sizeof(struct StatelessTsetlinMachine));
	if (sltm == NULL) {
		perror("Memory allocation failed");
		return NULL;
	}

	sltm->num_classes = num_classes;
	sltm->threshold = threshold;
	sltm->num_literals = num_literals;
	sltm->num_clauses = num_clauses;
	sltm->max_state = max_state;
	sltm->min_state = min_state;
	sltm->boost_true_positive_feedback = boost_true_positive_feedback;
	sltm->s = s;

	sltm->y_size = y_size;
	sltm->y_element_size = y_element_size;
	sltm->y_eq = sltm_y_eq_generic;
	sltm->output_activation = sltm_oa_class_idx;

	sltm->ta_state = (struct TANode **)malloc(
		num_clauses * sizeof(struct TANode *)); // shape: flat (num_clauses)
	if (sltm->ta_state == NULL) {
		perror("Memory allocation failed");
		sltm_free(sltm);
		return NULL;
	}
	for (uint32_t clause_id = 0; clause_id < sltm->num_clauses; clause_id++) {
		sltm->ta_state[clause_id] = NULL;
	}

	sltm->weights =
		(int16_t *)malloc(num_clauses * num_classes *
						  sizeof(int16_t)); // shape: flat (num_clauses, num_classes)
	if (sltm->weights == NULL) {
		perror("Memory allocation failed");
		sltm_free(sltm);
		return NULL;
	}

	sltm->clause_output =
		(uint8_t *)malloc(num_clauses * sizeof(uint8_t)); // shape: (num_clauses)
	if (sltm->clause_output == NULL) {
		perror("Memory allocation failed");
		sltm_free(sltm);
		return NULL;
	}

	sltm->votes =
		(int32_t *)malloc(num_classes * sizeof(int32_t)); // shape: (num_classes)
	if (sltm->votes == NULL) {
		perror("Memory allocation failed");
		sltm_free(sltm);
		return NULL;
	}

	sltm_initialize(sltm);

	return sltm;
}

/**
 * @brief Convert automaton state to action bit (include/exclude literal).
 */
static inline uint8_t action(int8_t state, int8_t mid_state) {
	return state >= mid_state;
}

/**
 * @brief Load a stateless machine by reading a dense TM binary and pruning.
 * @param filename Path to binary file created by a dense TM save.
 * @param y_size Output vector size per-row.
 * @param y_element_size Output element size in bytes.
 * @return Newly allocated StatelessTsetlinMachine or NULL on error.
 */
struct StatelessTsetlinMachine *sltm_load_dense(const char *filename, uint32_t y_size,
												uint32_t y_element_size) {
	FILE *file = fopen(filename, "rb");
	if (!file) {
		perror("Error opening file");
		return NULL;
	}

	uint32_t threshold, num_literals, num_clauses, num_classes;
	int8_t max_state, min_state;
	uint8_t boost_true_positive_feedback;
	double s_double;

	size_t threshold_read, num_literals_read, num_clauses_read, num_classes_read;
	size_t max_state_read, min_state_read, boost_true_positive_feedback_read,
		s_double_read;

	// Read metadata
	threshold_read = fread(&threshold, sizeof(uint32_t), 1, file);
	num_literals_read = fread(&num_literals, sizeof(uint32_t), 1, file);
	num_clauses_read = fread(&num_clauses, sizeof(uint32_t), 1, file);
	num_classes_read = fread(&num_classes, sizeof(uint32_t), 1, file);
	max_state_read = fread(&max_state, sizeof(int8_t), 1, file);
	min_state_read = fread(&min_state, sizeof(int8_t), 1, file);
	boost_true_positive_feedback_read =
		fread(&boost_true_positive_feedback, sizeof(uint8_t), 1, file);
	s_double_read = fread(&s_double, sizeof(double), 1, file);

	if (threshold_read != 1 || num_literals_read != 1 || num_clauses_read != 1 ||
		num_classes_read != 1 || max_state_read != 1 || min_state_read != 1 ||
		boost_true_positive_feedback_read != 1 || s_double_read != 1) {
		fprintf(stderr, "Failed to read all metadata from bin\n");
		fclose(file);
		return NULL;
	}

	struct StatelessTsetlinMachine *sltm = sltm_create(
		num_classes, threshold, num_literals, num_clauses, max_state, min_state,
		boost_true_positive_feedback, y_size, y_element_size, (float)s_double);
	if (!sltm) {
		fprintf(stderr, "sltm_create failed\n");
		fclose(file);
		return NULL;
	}

	// Allocate and read weights
	size_t weights_read =
		fread(sltm->weights, sizeof(int16_t), num_clauses * num_classes, file);
	if (weights_read != num_clauses * num_classes) {
		fprintf(stderr, "Failed to read all weights from bin\n");
		sltm_free(sltm);
		fclose(file);
		return NULL;
	}
	// Allocate and read clauses
	int8_t *flat_states = malloc(num_clauses * num_literals * 2 * sizeof(int8_t));
	size_t states_read =
		fread(flat_states, sizeof(int8_t), num_clauses * num_literals * 2, file);
	if (states_read != num_clauses * num_literals * 2) {
		fprintf(stderr, "Failed to read all states from bin\n");
		sltm_free(sltm);
		free(flat_states);
		fclose(file);
		return NULL;
	}

	sltm_free_state_llists(sltm);
	for (uint32_t clause_id = 0; clause_id < sltm->num_clauses; clause_id++) {
		struct TANode *prev_ptr = NULL;
		struct TANode **head_ptr_addr = sltm->ta_state + clause_id;

		for (uint32_t i = 0; i < sltm->num_literals * 2; i++) {
			if (action(flat_states[clause_id * sltm->num_literals * 2 + i],
					   sltm->mid_state)) {
				ta_stateless_insert(head_ptr_addr, prev_ptr, i, &prev_ptr);
			}
		}
	}
	free(flat_states);

	fclose(file);
	return sltm;
}

/**
 * @brief Save a stateless machine to a binary file.
 */
void sltm_save(const struct StatelessTsetlinMachine *sltm, const char *filename) {
	FILE *file = fopen(filename, "wb");
	if (!file) {
		perror("Error opening file for writing");
		return;
	}

	size_t written;

	written = fwrite(&sltm->threshold, sizeof(uint32_t), 1, file);
	if (written != 1) {
		fprintf(stderr, "Failed to write threshold\n");
		goto save_error;
	}
	written = fwrite(&sltm->num_literals, sizeof(uint32_t), 1, file);
	if (written != 1) {
		fprintf(stderr, "Failed to write num_literals\n");
		goto save_error;
	}
	written = fwrite(&sltm->num_clauses, sizeof(uint32_t), 1, file);
	if (written != 1) {
		fprintf(stderr, "Failed to write num_clauses\n");
		goto save_error;
	}
	written = fwrite(&sltm->num_classes, sizeof(uint32_t), 1, file);
	if (written != 1) {
		fprintf(stderr, "Failed to write num_classes\n");
		goto save_error;
	}
	written = fwrite(&sltm->max_state, sizeof(int8_t), 1, file);
	if (written != 1) {
		fprintf(stderr, "Failed to write max_state\n");
		goto save_error;
	}
	written = fwrite(&sltm->min_state, sizeof(int8_t), 1, file);
	if (written != 1) {
		fprintf(stderr, "Failed to write min_state\n");
		goto save_error;
	}
	written = fwrite(&sltm->boost_true_positive_feedback, sizeof(uint8_t), 1, file);
	if (written != 1) {
		fprintf(stderr, "Failed to write boost_true_positive_feedback\n");
		goto save_error;
	}
	written = fwrite(&sltm->s, sizeof(double), 1, file);
	if (written != 1) {
		fprintf(stderr, "Failed to write s parameter\n");
		goto save_error;
	}
	size_t n_weights = (size_t)sltm->num_clauses * sltm->num_classes;
	written = fwrite(sltm->weights, sizeof(int16_t), n_weights, file);
	if (written != n_weights) {
		fprintf(stderr, "Failed to write weights array (%zu of %zu)\n", written,
				n_weights);
		goto save_error;
	}
	for (uint32_t clause_id = 0; clause_id < sltm->num_clauses; clause_id++) {
		struct TANode *curr_ptr = sltm->ta_state[clause_id];
		while (curr_ptr != NULL) {
			written = fwrite(&curr_ptr->ta_id, sizeof(uint32_t), 1, file);
			if (written != 1) {
				fprintf(stderr, "Failed to write node ta_id\n");
				goto save_error;
			}
			curr_ptr = curr_ptr->next;
		}
		uint32_t delim = UINT_MAX;
		written = fwrite(&delim, sizeof(uint32_t), 1, file);
		if (written != 1) {
			fprintf(stderr, "Failed to write delimiter\n");
			goto save_error;
		}
	}

	fclose(file);
	return;

save_error:
	fclose(file);
	fprintf(stderr, "sltm_save aborted, file %s may be incomplete\n", filename);
}

/**
 * @brief Free all state linked-lists held by a stateless machine.
 */
inline static void sltm_free_state_llists(struct StatelessTsetlinMachine *sltm) {
	for (uint32_t clause_id = 0; clause_id < sltm->num_clauses; clause_id++) {
		struct TANode **head_ptr = sltm->ta_state + clause_id;
		while (*head_ptr != NULL) {
			ta_stateless_remove(head_ptr, NULL, NULL);
		}
	}
}

/**
 * @brief Free a StatelessTsetlinMachine and its resources.
 */
void sltm_free(struct StatelessTsetlinMachine *sltm) {
	if (sltm != NULL) {
		if (sltm->ta_state != NULL) {
			sltm_free_state_llists(sltm);
			free(sltm->ta_state);
			sltm->ta_state = NULL;
		}

		if (sltm->weights != NULL) {
			free(sltm->weights);
			sltm->weights = NULL;
		}

		if (sltm->clause_output != NULL) {
			free(sltm->clause_output);
			sltm->clause_output = NULL;
		}

		if (sltm->votes != NULL) {
			free(sltm->votes);
			sltm->votes = NULL;
		}

		free(sltm);
	}

	return;
}

/**
 * @brief Initialize derived fields for a stateless machine (mid_state, s_inv, etc.).
 */
void sltm_initialize(struct StatelessTsetlinMachine *sltm) {
	sltm->mid_state = (sltm->max_state + sltm->min_state) / 2;
	sltm->s_inv = 1.0f / sltm->s;
	sltm->s_min1_inv = (sltm->s - 1.0f) / sltm->s;
}

/**
 * @brief Compute clause outputs for a single input row (stateless machine).
 */
static inline void calculate_clause_output(struct StatelessTsetlinMachine *sltm,
										   const uint8_t *X) {
	// For each clause, check if it is "active" - all necessary literals have the right
	// value
	for (uint32_t clause_id = 0; clause_id < sltm->num_clauses; clause_id++) {
		sltm->clause_output[clause_id] = 1;
		uint8_t empty_clause = 1;

		// Clause is active if:
		// - it's not empty (unless skip_empty is unset as should be the case for
		// training)
		// - each literal present in the clause has the right value (same as the input X)
		// Iterate over linked list of Tsetlin Automata ids
		struct TANode *curr_ptr = sltm->ta_state[clause_id];
		while (curr_ptr != NULL) {
			empty_clause = 0;
			if (curr_ptr->ta_id % 2 == X[curr_ptr->ta_id / 2]) {
				sltm->clause_output[clause_id] = 0;
				break;
			}
			curr_ptr = curr_ptr->next;
		}
		if (empty_clause) {
			sltm->clause_output[clause_id] = 0;
		}
	}
}

/**
 * @brief Sum clause votes into class votes and clip to threshold (stateless).
 */
static inline void sum_votes(struct StatelessTsetlinMachine *sltm) {
	memset(sltm->votes, 0, sltm->num_classes * sizeof(int32_t));

	// Simple sum of votes for each class, then clip them to the threshold
	for (uint32_t clause_id = 0; clause_id < sltm->num_clauses; clause_id++) {
		if (sltm->clause_output[clause_id] == 0) {
			continue;
		}

		for (uint32_t class_id = 0; class_id < sltm->num_classes; class_id++) {
			sltm->votes[class_id] +=
				sltm->weights[(clause_id * sltm->num_classes) + class_id];
		}
	}

	for (uint32_t class_id = 0; class_id < sltm->num_classes; class_id++) {
		sltm->votes[class_id] = clip(sltm->votes[class_id], (int32_t)sltm->threshold);
	}
}

/**
 * @brief Run inference with a stateless machine.
 * @param sltm Pointer to the stateless machine.
 * @param X Input data buffer (rows x num_literals).
 * @param y_pred Preallocated output buffer.
 * @param rows Number of input rows.
 */
void sltm_predict(struct StatelessTsetlinMachine *sltm, const uint8_t *X, void *y_pred,
				  uint32_t rows) {
	for (uint32_t row = 0; row < rows; row++) {
		const uint8_t *X_row = X + (row * sltm->num_literals);
		void *y_pred_row =
			(void *)(((uint8_t *)y_pred) + (row * sltm->y_size * sltm->y_element_size));

		// Calculate clause output
		calculate_clause_output(sltm, X_row);

		// Sum up clause votes for each class
		sum_votes(sltm);

		// Pass through output activation function
		sltm->output_activation(sltm, y_pred_row);
	}
}

/**
 * @brief Evaluate predictions on a dataset and print accuracy (stateless).
 */
void sltm_evaluate(struct StatelessTsetlinMachine *sltm, const uint8_t *X, const void *y,
				   uint32_t rows) {
	uint32_t correct = 0;
	uint32_t total = 0;
	void *y_pred = malloc(rows * sltm->y_size * sltm->y_element_size);
	if (y_pred == NULL) {
		perror("Memory allocation failed\n");
		exit(1);
	}

	sltm_predict(sltm, X, y_pred, rows);

	for (uint32_t row = 0; row < rows; ++row) {

		void *y_row =
			(void *)(((uint8_t *)y) + (row * sltm->y_size * sltm->y_element_size));
		void *y_pred_row =
			(void *)(((uint8_t *)y_pred) + (row * sltm->y_size * sltm->y_element_size));

		if (sltm->y_eq(sltm, y_row, y_pred_row)) {
			correct++;
		}
		total++;
	}
	printf("correct: %d, total: %d, ratio: %.2f \n", correct, total,
		   (float)correct / total);
	free(y_pred);
}

// --- Basic output_activation functions ---

/**
 * @brief Output activation returning the class index with highest vote.
 */
void sltm_oa_class_idx(const struct StatelessTsetlinMachine *sltm, const void *y_pred) {
	if (sltm->y_size != 1) {
		fprintf(stderr, "y_eq_class_idx expects y_size == 1");
		exit(1);
	}
	uint32_t *label_pred = (uint32_t *)y_pred;

	// class index compare
	uint32_t best_class = 0;
	int32_t max_class_score = sltm->votes[0];
	for (uint32_t class_id = 1; class_id < sltm->num_classes; class_id++) {
		if (max_class_score < sltm->votes[class_id]) {
			max_class_score = sltm->votes[class_id];
			best_class = class_id;
		}
	}

	*label_pred = best_class;
}

/**
 * @brief Output activation producing a per-class binary vector using mid_state.
 */
void sltm_oa_bin_vector(const struct StatelessTsetlinMachine *sltm, const void *y_pred) {
	if (sltm->y_size != sltm->num_classes) {
		fprintf(stderr, "y_eq_bin_vector expects y_size == tm->num_classes");
		exit(1);
	}
	uint8_t *y_bin_vec = (uint8_t *)y_pred;

	for (uint32_t class_id = 0; class_id < sltm->num_classes; class_id++) {
		// binary threshold (k=mid_state)
		y_bin_vec[class_id] = (sltm->votes[class_id] > sltm->mid_state);
	}
}

// Set the output activation function for the Tsetlin Machine
void sltm_set_output_activation(
	struct StatelessTsetlinMachine *sltm,
	void (*output_activation)(const struct StatelessTsetlinMachine *sltm,
							  const void *y_pred)) {
	sltm->output_activation = output_activation;
}
