#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "FASTA.h"

#define VERSION "0.2.0"

#define THREADS_PER_BLOCK 512
#define BLOCKS_PER_GRID 512
#define ALPHA 4

typedef struct {
    int match;
    int mismatch;
    int first;
    int extension;
} scoring;

typedef struct {
    int score;
    int row;
    int column;
} alignmentScore;

typedef struct {
	int H;
	int E;
	int F;
} element;

__global__ void kernel1(int dk) {

}

__global__ void kernel2(int dk) {

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
    values.first = atoi(argv[5]);
    values.extension = atoi(argv[6]);

	if(values.match < 1 || values.mismatch > -1 || values.first > -1 || values.extension > -1) {
		printf("ERROR\nOne or more scoring values were not usable!\n\n");
		exit(-1);
	}
	else {
		printf("DONE\n\nScoring values:\n");
		printf("	>Match: %d\n", values.match);
		printf("	>Mismatch: %d\n", values.mismatch);
		printf("	>First gap: %d\n", values.first);
		printf("	>Gap extension: %d\n\n", values.extension);
	}

    printf("Starting alignment process... ");

    element *matrix;
    int matrixSize = sizeof(element) * (first.getLength() + 1) * (second.getLength() + 1);
    cudaMalloc(&matrix, matrixSize);
    cudaMemset(matrix, 0, matrixSize);

    alignmentScore max;
    max.score = 0;

    alignmentScore *blockScores;
    int blockScoreSize = sizeof(alignmentScore) * BLOCKS_PER_GRID;
    blockScores = (alignmentScore *)malloc(blockScoreSize);
    alignmentScore *devBlockScore;
    cudaMalloc(&devBlockScore, blockScoreSize);
    cudaMemset(devBlockScore, 0, blockScoreSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int D = BLOCKS_PER_GRID + first.getLength() / (ALPHA * THREADS_PER_BLOCK) - 1;
    for(int dk = 0; dk < D + BLOCKS_PER_GRID; dk++) {
    	kernel1<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dk);
    	cudaDeviceSynchronize();
    	kernel2<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dk);

    	cudaMemcpy(blockScores, devBlockScore, blockScoreSize, cudaMemcpyDeviceToHost);
    	for(int i = 0; i < BLOCKS_PER_GRID; i++) {
    		if(max.score < blockScores[i].score) {
    			max.score = blockScores[i].score;
    			max.column = blockScores[i].column;
    			max.row = blockScores[i].row;
    		}
    	}
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    printf("DONE\n\n");

    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Kernel executed in %f s\n", time / 1000);

    printf("\nAlignment score: %d at [%d,%d]\n", max.score, max.row, max.column);

    cudaFree(matrix);
    cudaFree(devBlockScore);

    free(blockScores);

    return 0;
}
