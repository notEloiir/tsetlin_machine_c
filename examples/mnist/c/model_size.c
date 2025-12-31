#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "sparse_tsetlin_machine.h"
#include "stateless_tsetlin_machine.h"
#include "tsetlin_machine.h"

void print_fsize(const char *filename) {
	struct stat st;

	if (stat(filename, &st) == 0) {
		printf("%s size: %ld\n", filename, st.st_size);
	}
}

int main() {
	const char *file_path = "examples/mnist/data/mnist_tm.bin";
	print_fsize(file_path);

	struct TsetlinMachine *tm = tm_load(file_path, 1, sizeof(int32_t));
	if (tm == NULL) {
		perror("tm_load failed");
		return 1;
	}
	tm_save(tm, "examples/mnist/data/dense.bin");
	tm_free(tm);
	print_fsize("examples/mnist/data/dense.bin");

	struct SparseTsetlinMachine *stm = stm_load_dense(file_path, 1, sizeof(int32_t));
	if (stm == NULL) {
		perror("stm_load_dense failed");
		return 1;
	}
	stm_save(stm, "examples/mnist/data/sparse.bin");
	stm_free(stm);
	print_fsize("examples/mnist/data/sparse.bin");

	struct StatelessTsetlinMachine *sltm = sltm_load_dense(file_path, 1, sizeof(int32_t));
	if (sltm == NULL) {
		perror("sltm_load_dense failed");
		return 1;
	}
	sltm_save(sltm, "examples/mnist/data/stateless.bin");
	sltm_free(sltm);
	print_fsize("examples/mnist/data/stateless.bin");

	return 0;
}
