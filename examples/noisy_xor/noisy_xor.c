// cmake --build build && ./build/examples/noisy_xor/noisy_xor
#include <stdio.h>

#include "data/noisy_xor_dataset.h"
#include "tsetlin_machine.h"

int main() {
  // Define parameters
  uint32_t num_classes = 2;
  uint32_t threshold = 1000;
  uint32_t num_literals = NOISY_XOR_FEATURES;
  uint32_t num_clauses = 1000;
  int8_t max_state = 127;
  int8_t min_state = -127;
  uint8_t boost_true_positive_feedback = 0;  // False
  uint32_t y_size = 1;
  uint32_t y_element_size = sizeof(typeof(y_data[0]));
  float s = 3.0f;
  int epochs = 10;
  uint32_t seed = 42;

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
  tm_train(tm, X_train, y_train, train_samples, epochs);

  // Evaluate the model
  tm_evaluate(tm, X_test, y_test, test_samples);

  // Clean up
  tm_free(tm);

  return 0;
}