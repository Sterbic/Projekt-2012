#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "FASTA.h"

#define MAXLEN 5005
#define VERSION "0.2.0"

typedef struct {
    int match;
    int mismatch;
    int insertion;
    int deletion;
} scoring;

typedef struct {
    int score;
    int row;
    int column;
} entry;

__global__ void kernel() {

}

int main(int argc, char *argv[]) {
	printf("> Welcome to SWaligner v%s\n\n", VERSION);

	if (argc != 7) {
        printf("Expected 6 input arguments, not %d!\n\n", argc - 1);
        return -1;
    }

	FASTAsequence first(argv[1]);
	FASTAsequence second(argv[2]);

	printf("Loading input sequences... ");
	if(!first.load() || !second.load()) {
		printf("ERROR\nAn error has occured while loading input sequences!\n\n");
		exit(-1);
	}
	else {
		printf("DONE\n\n");
	}

	printf("First sequence:\n%s\n\n", first.getSequenceName());
	printf("Second sequence:\n%s\n\n", second.getSequenceName());

	scoring values;
	printf("Initializing scoring values... ");
    values.match = atoi(argv[3]);
    values.mismatch = atoi(argv[4]);
    values.insertion = atoi(argv[5]);
    values.deletion = atoi(argv[6]);

	if(values.match < 1 || values.mismatch > -1 || values.insertion > -1 || values.deletion > -1) {
		printf("ERROR\nOne or more scoring values were not usable!\n\n");
		exit(-1);
	}
	else {
		printf("DONE\n\nScoring values:\n");
		printf("	>Match: %d\n", values.match);
		printf("	>Mismatch: %d\n", values.mismatch);
		printf("	>Insertion: %d\n", values.insertion);
		printf("	>Deletion: %d\n\n", values.deletion);
	}

	printf("Starting alignment process... ");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	kernel<<<10, 10>>>();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	printf("DONE\n\n");

	float time;
	cudaEventElapsedTime(&time, start, stop);
	printf("Kernel executed in %f s\n", time / 1000);

    return 0;
}
