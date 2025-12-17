// cmake --build build && ./build/examples/noisy_xor/noisy_xor_benchmark <<<
// "1000 1000 127 -127 0 3.0 10 42"
#include <stdio.h>
#include <time.h>

#include "data/noisy_xor_dataset.h"
#include "tsetlin_machine.h"

void train_model(struct TsetlinMachine* tm, const uint8_t* X_train,
                 const uint32_t* y_train, uint32_t train_samples, int epochs) {
  clock_t start_train = clock();
  tm_train(tm, X_train, y_train, train_samples, epochs);
  clock_t end_train = clock();
  double time_spent_train = (double)(end_train - start_train) / CLOCKS_PER_SEC;
  printf("Training time: %f seconds\n", time_spent_train);
}

void evaluate_model(struct TsetlinMachine* tm, const uint8_t* X_test,
                    const uint32_t* y_test, uint32_t test_samples) {
  clock_t start_eval = clock();
  tm_evaluate(tm, X_test, y_test, test_samples);
  clock_t end_eval = clock();
  double time_spent_eval = (double)(end_eval - start_eval) / CLOCKS_PER_SEC;
  printf("Evaluation time: %f seconds\n", time_spent_eval);
}

int main() {
  // Define parameters
  uint32_t num_classes = 2;
  uint32_t threshold;  // to be read from stdin
  uint32_t num_literals = NOISY_XOR_FEATURES;
  uint32_t num_clauses;                  // to be read from stdin
  int8_t max_state;                      // to be read from stdin
  int8_t min_state;                      // to be read from stdin
  uint8_t boost_true_positive_feedback;  // to be read from stdin
  uint32_t y_size = 1;
  uint32_t y_element_size = sizeof(typeof(y_data[0]));
  float s;        // to be read from stdin
  int epochs;     // to be read from stdin
  uint32_t seed;  // to be read from stdin

  // Read parameters from stdin
  if (scanf("%u %u %hhd %hhd %hhu %f %d %u", &threshold, &num_clauses,
            &max_state, &min_state, &boost_true_positive_feedback, &s, &epochs,
            &seed) != 8) {
    fprintf(stderr, "Error reading parameters from stdin\n");
    return 1;
  }

  printf("Parameters:\n");
  printf("  num_classes: %u\n", num_classes);
  printf("  threshold: %u\n", threshold);
  printf("  num_literals: %u\n", num_literals);
  printf("  num_clauses: %u\n", num_clauses);
  printf("  max_state: %d\n", max_state);
  printf("  min_state: %d\n", min_state);
  printf("  boost_true_positive_feedback: %u\n", boost_true_positive_feedback);
  printf("  s: %f\n", s);
  printf("  epochs: %d\n", epochs);
  printf("  seed: %u\n", seed);

  // Create Tsetlin Machine
  struct TsetlinMachine* tm = tm_create(
      num_classes, threshold, num_literals, num_clauses, max_state, min_state,
      boost_true_positive_feedback, y_size, y_element_size, s, seed);
  if (tm == NULL) {
    perror("tm_create failed");
    return 1;
  }

  // Split data
  uint32_t train_samples = (uint32_t)(NOISY_XOR_SAMPLES * 0.8);
  uint32_t test_samples = NOISY_XOR_SAMPLES - train_samples;

  const uint8_t* X_train = X_data;
  const uint32_t* y_train = y_data;

  const uint8_t* X_test = X_data + train_samples * NOISY_XOR_FEATURES;
  const uint32_t* y_test = y_data + train_samples;

  // Train the model
  train_model(tm, X_train, y_train, train_samples, epochs);

  // Evaluate the model
  evaluate_model(tm, X_test, y_test, test_samples);

  // Clean up
  tm_free(tm);

  return 0;
}