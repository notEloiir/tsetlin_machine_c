#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// Default values
#define DEFAULT_N_SAMPLES 5000
#define DEFAULT_N_FEATURES 12
#define DEFAULT_N_XOR_FEATURES 2
#define DEFAULT_NOISE_LEVEL 0.1
#define DEFAULT_SEED 42

// Default output path relative to where the executable is run
#define OUTPUT_FILENAME "examples/noisy_xor/data/noisy_xor_dataset.h"

int main(int argc, char *argv[]) {
    const char *filename = OUTPUT_FILENAME;
    int n_samples = DEFAULT_N_SAMPLES;
    int n_features = DEFAULT_N_FEATURES;
    int n_xor_features = DEFAULT_N_XOR_FEATURES;
    float noise_level = DEFAULT_NOISE_LEVEL;
    int seed = DEFAULT_SEED;

    if (argc >= 6) {
        n_samples = atoi(argv[1]);
        n_features = atoi(argv[2]);
        n_xor_features = atoi(argv[3]);
        noise_level = (float)atof(argv[4]);
        seed = atoi(argv[5]);
        
        if (argc >= 7) {
            filename = argv[6];
        }
    } else if (argc == 2) {
        filename = argv[1];
    }

    // Use fixed seed for reproducibility
    srand(seed);

    uint8_t *X = (uint8_t *)malloc(n_samples * n_features * sizeof(uint8_t));
    uint32_t *y = (uint32_t *)malloc(n_samples * sizeof(uint32_t));

    if (!X || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Generate random binary features and calculate clean XOR
    for (int i = 0; i < n_samples; i++) {
        uint32_t xor_result = 0;
        for (int j = 0; j < n_features; j++) {
            int bit = rand() % 2;
            X[i * n_features + j] = bit;
            
            if (j < n_xor_features) {
                if (j == 0) {
                    xor_result = bit;
                } else {
                    xor_result ^= bit;
                }
            }
        }
        y[i] = xor_result;
    }

    // Introduce noise
    int n_noise = (int)(noise_level * n_samples);
    
    // Create an array of indices to shuffle for selecting noise targets without replacement
    int *indices = (int *)malloc(n_samples * sizeof(int));
    if (!indices) {
        fprintf(stderr, "Memory allocation failed for indices\n");
        free(X);
        free(y);
        return 1;
    }
    for (int i = 0; i < n_samples; i++) {
        indices[i] = i;
    }

    // Fisher-Yates shuffle (partial) to pick unique indices
    for (int i = 0; i < n_noise; i++) {
        int j = i + rand() % (n_samples - i);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
        
        // Flip the bit at the selected index
        y[indices[i]] = 1 - y[indices[i]];
    }
    free(indices);

    // Write to header file
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file %s for writing. Ensure the directory exists.\n", filename);
        free(X);
        free(y);
        return 1;
    }

    fprintf(fp, "/**\n");
    fprintf(fp, " * Auto-generated Noisy XOR dataset.\n");
    fprintf(fp, " * Samples: %d, Features: %d, XOR Features: %d, Noise: %.2f, Seed: %d\n", 
            n_samples, n_features, n_xor_features, noise_level, seed);
    fprintf(fp, " */\n\n");
    
    fprintf(fp, "#ifndef NOISY_XOR_DATASET_H\n");
    fprintf(fp, "#define NOISY_XOR_DATASET_H\n\n");
    fprintf(fp, "#include <stdint.h>\n\n");
    
    fprintf(fp, "#define NOISY_XOR_SAMPLES %d\n", n_samples);
    fprintf(fp, "#define NOISY_XOR_FEATURES %d\n\n", n_features);

    fprintf(fp, "static const uint8_t X_data[] = {\n");
    for (int i = 0; i < n_samples * n_features; i++) {
        fprintf(fp, "%d", X[i]);
        if (i < n_samples * n_features - 1) {
            fprintf(fp, ",");
        }
        if ((i + 1) % n_features == 0) {
            fprintf(fp, "\n");
        } else {
             fprintf(fp, " ");
        }
    }
    fprintf(fp, "};\n\n");

    fprintf(fp, "static const uint32_t y_data[] = {\n");
    for (int i = 0; i < n_samples; i++) {
        fprintf(fp, "%d", y[i]);
        if (i < n_samples - 1) {
            fprintf(fp, ", ");
            if ((i + 1) % 20 == 0) {
                fprintf(fp, "\n");
            }
        }
    }
    fprintf(fp, "\n};\n\n");

    fprintf(fp, "#endif // NOISY_XOR_DATASET_H\n");

    fclose(fp);
    free(X);
    free(y);

    printf("Dataset generated at %s\n", filename);

    return 0;
}
