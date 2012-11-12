#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>

#include "FASTA.h"

#define VERSION "0.2.0"

#define THREADS_PER_BLOCK 9
#define BLOCKS_PER_GRID 1
#define ALPHA 2

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

__global__ void shortPhase(int dk, element *matrix,char *first,
	int firstLength, char *second, int secondLength, scoring values,
	alignmentScore *blockScore) {
    
    printf("in short phase thread %d\n", threadIdx.x);

	__shared__ alignmentScore blockMax[THREADS_PER_BLOCK];
	blockMax[threadIdx.x].score = 0;

	int i = ALPHA * (blockIdx.x * THREADS_PER_BLOCK + threadIdx.x);
	int j = secondLength / BLOCKS_PER_GRID * (dk - blockIdx.x) - threadIdx.x;
	printf("blockIdx.x = %d, threadIdx.x = %d, i = %d, j = %d\n", blockIdx.x, threadIdx.x, i, j);

	if(j < 0) {
		i -= ALPHA * BLOCKS_PER_GRID * THREADS_PER_BLOCK;
		j = secondLength - j;
	}

	for(int innerDiagonal = 0; innerDiagonal < THREADS_PER_BLOCK; innerDiagonal++) {
		if(i >= 0 && i < firstLength) {
			for(int a = 0; a < ALPHA; a++) {
				element *current = matrix + j + 1 + (i + a + 1) * (secondLength + 1);
				element *left = matrix + j + (i + a + 1) * (secondLength + 1);
				element *up = matrix + j + 1 + (i + a) * (secondLength + 1);
				element *diagonal = matrix + j + (i + a) * (secondLength + 1);

				int matchMissmatch = values.mismatch;
				if(first[i + a] == second[j]) matchMissmatch = values.match;

				current->E = max(left->E + values.extension, left->H + values.first);
				current->F = max(up->F + values.extension, up->H + values.first);
				current->H = max(max(0, current->E), max(current->F, diagonal->H + matchMissmatch));

				if(current->H > blockMax[threadIdx.x].score) {
					blockMax[threadIdx.x].score = current->H;
					blockMax[threadIdx.x].column = j + 1;
					blockMax[threadIdx.x].row = i + 1;
				}
			}
		}

		j++;

		if(j == secondLength) {
			j = 0;
			i += BLOCKS_PER_GRID * ALPHA * THREADS_PER_BLOCK;
		}

		__syncthreads();
	}

	int index = blockDim.x / 2;
	while(i != 0) {
		if(threadIdx.x < index) {
			if(blockMax[threadIdx.x].score < blockMax[threadIdx.x + index].score)
				blockMax[threadIdx.x] = blockMax[threadIdx.x + index];
		}

		__syncthreads();
		index /= 2;
	}

	if(threadIdx.x == 0) {
		blockScore[blockIdx.x] = blockMax[0];
	}
}

__global__ void longPhase(int dk, element *matrix,char *first,
		int firstLength, char *second, int secondLength, scoring values,
		alignmentScore *blockScore) {
	__shared__ alignmentScore blockMax[THREADS_PER_BLOCK];
	blockMax[threadIdx.x].score = 0;

	int i = ALPHA * (blockIdx.x * THREADS_PER_BLOCK + threadIdx.x);
	int j = secondLength / BLOCKS_PER_GRID * (dk - blockIdx.x) - threadIdx.x;

	if(j < 0) {
		i -= ALPHA * BLOCKS_PER_GRID * THREADS_PER_BLOCK;
		j = secondLength - j;
	}

	for(int innerDiagonal = THREADS_PER_BLOCK; innerDiagonal < secondLength / BLOCKS_PER_GRID; innerDiagonal++) {
		if(i >= 0 && i < firstLength) {
			for(int a = 0; a < ALPHA; a++) {
				element *current = matrix + j + 1 + (i + a + 1) * (secondLength + 1);
				element *left = matrix + j + (i + a + 1) * (secondLength + 1);
				element *up = matrix + j + 1 + (i + a) * (secondLength + 1);
				element *diagonal = matrix + j + (i + a) * (secondLength + 1);

				int matchMissmatch = values.mismatch;
				if(first[i + a] == second[j]) matchMissmatch = values.match;

				current->E = max(left->E + values.extension, left->H + values.first);
				current->F = max(up->F + values.extension, up->H + values.first);
				current->H = max(max(0, current->E), max(current->F, diagonal->H + matchMissmatch));

				if(current->H > blockMax[threadIdx.x].score) {
					blockMax[threadIdx.x].score = current->H;
					blockMax[threadIdx.x].column = j + 1;
					blockMax[threadIdx.x].row = i + 1;
				}
			}
		}

		j++;

		if(j == secondLength) {
			j = 0;
			i += BLOCKS_PER_GRID * ALPHA * THREADS_PER_BLOCK;
		}

		__syncthreads();
	}

	int index = blockDim.x / 2;
	while(i != 0) {
		if(threadIdx.x < index) {
			if(blockMax[threadIdx.x].score < blockMax[threadIdx.x + index].score)
				blockMax[threadIdx.x] = blockMax[threadIdx.x + index];
		}

		__syncthreads();
		index /= 2;
	}

	if(threadIdx.x == 0) {
		blockScore[blockIdx.x] = blockMax[0];
	}
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
    else
      printf("DONE\n\n");
	
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

    char *devFirst;
    cudaMalloc(&devFirst, first.getLength() * sizeof(char));
    cudaMemcpy(devFirst, first.getSequence(), first.getLength() * sizeof(char), cudaMemcpyHostToDevice);

    char *devSecond;
    cudaMalloc(&devSecond, second.getLength() * sizeof(char));
    cudaMemcpy(devSecond, second.getSequence(), second.getLength() * sizeof(char), cudaMemcpyHostToDevice);

	int D = BLOCKS_PER_GRID + first.getLength() / (ALPHA * THREADS_PER_BLOCK) - 1;
	
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    for(int dk = 0; dk < D + BLOCKS_PER_GRID; dk++) {
    	shortPhase<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    			dk,
    			matrix,
    			devFirst,
    			first.getLength(),
    			devSecond,
    			second.getLength(),
    			values,
    			devBlockScore
    			);

        printf("\nLaunch %d, error: %s\n", dk, cudaGetErrorString(cudaDeviceSynchronize()));
    	
    	longPhase<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    			dk,
    			matrix,
    			devFirst,
    			first.getLength(),
    			devSecond,
    			second.getLength(),
    			values,
    			devBlockScore
    			);

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
    cudaFree(devFirst);
    cudaFree(devSecond);

    free(blockScores);

    printf("\n%s\n", cudaGetErrorString(cudaGetLastError()));

    return 0;
}
