#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>

#include "FASTA.h"

#define VERSION "0.2.5"

#define THREADS_PER_BLOCK 9
#define BLOCKS_PER_GRID 1
#define ALPHA 1

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

__global__ void shortPhase(int dk, element *matrix, char *first,
	int firstLength, char *second, int secondLength, scoring values,
	alignmentScore *blockScore) {
	__shared__ alignmentScore blockMax[THREADS_PER_BLOCK];
	blockMax[threadIdx.x].score = 0;

	int i = ALPHA * (blockIdx.x * THREADS_PER_BLOCK + threadIdx.x);
	int j = secondLength / BLOCKS_PER_GRID * (dk - blockIdx.x) - threadIdx.x;

	if(j < 0) {
		i -= ALPHA * BLOCKS_PER_GRID * THREADS_PER_BLOCK;
		j = secondLength + j;
	}

//	printf("blockIdx.x = %d, threadIdx.x = %d, i = %d, j = %d\n", blockIdx.x, threadIdx.x, i, j);
	int cols = secondLength + 1;
	element *test = &matrix[6 + 6 * cols];

	for(int innerDiagonal = 0; innerDiagonal < THREADS_PER_BLOCK; innerDiagonal++) {
		if(j == secondLength) {
					j = 0;
					i += BLOCKS_PER_GRID * ALPHA * THREADS_PER_BLOCK;
		}

		if(i >= 0 && i < firstLength) {
			element *current = &matrix[j + 1 + (i + 1) * cols];
			element *left = current - 1;
			element *up = current - cols;
			element *diagonal = up - 1;

			printf("[%d, %d] = [%d, %d, %d] -- old\n", i, j, current->H, current->E, current->F);
			if(current == test) printf("Pogodak u shortu!!!\n");

			int matchMissmatch = values.mismatch;
			if(first[i] == second[j])
				matchMissmatch = values.match;

			current->E = max(left->E + values.extension, left->H + values.first);
			current->F = max(up->F + values.extension, up->H + values.first);
			current->H = max(max(0, current->E), max(current->F, diagonal->H + matchMissmatch));

		//	printf("[%d, %d] = [%d, %d, %d] -- new\n\n", i, j, current->H, current->E, current->F);

			if(current->H > blockMax[threadIdx.x].score) {
				blockMax[threadIdx.x].score = current->H;
				blockMax[threadIdx.x].row = i;
				blockMax[threadIdx.x].column = j;
			}
		}

		j++;

/*		if(j == secondLength) {
			j = 0;
			i += BLOCKS_PER_GRID * ALPHA * THREADS_PER_BLOCK;
		} */

		__syncthreads();
	}

	int nearestPowof2 = 1;
	while (nearestPowof2 < THREADS_PER_BLOCK)
		nearestPowof2 <<= 1;

	int index = nearestPowof2 / 2;
	while(index != 0) {
		if(threadIdx.x < index && threadIdx.x + index < THREADS_PER_BLOCK) {
			if(blockMax[threadIdx.x].score < blockMax[threadIdx.x + index].score)
				blockMax[threadIdx.x] = blockMax[threadIdx.x + index];
		}

		__syncthreads();
		index /= 2;
	}

	if(threadIdx.x == 0 && blockScore[blockIdx.x].score < blockMax[0].score) {
		blockScore[blockIdx.x] = blockMax[0];
		for(int i = 0; i < THREADS_PER_BLOCK; i++) {
			printf("%d ", blockMax[i].score);
		}
	}
}

__global__ void longPhase(int dk, element *matrix,char *first,
		int firstLength, char *second, int secondLength, scoring values,
		alignmentScore *blockScore) {
	__shared__ alignmentScore blockMax[THREADS_PER_BLOCK];
	blockMax[threadIdx.x].score = 0;

	int i = ALPHA * (blockIdx.x * THREADS_PER_BLOCK + threadIdx.x);
	int j = secondLength / BLOCKS_PER_GRID * (dk - blockIdx.x) - threadIdx.x + THREADS_PER_BLOCK;

	int cols = secondLength + 1;
	element *test = &matrix[6 + 6 * cols];

	for(int innerDiagonal = THREADS_PER_BLOCK;
			innerDiagonal < secondLength / BLOCKS_PER_GRID; innerDiagonal++) {
		if(i >= 0 && i < firstLength) {
			for(int a = 0; a < ALPHA; a++) {
				element *current = &matrix[j + 1 + (i + 1) * cols];
				element *left = current - 1;
				element *up = current - cols;
				element *diagonal = up - 1;
				if(current == test) printf("Pogodak u longu!!!\n");

				int matchMissmatch = values.mismatch;
				if(first[i] == second[j])
				matchMissmatch = values.match;

				current->E = max(left->E + values.extension, left->H + values.first);
				current->F = max(up->F + values.extension, up->H + values.first);
				current->H = max(max(0, current->E), max(current->F, diagonal->H + matchMissmatch));
				printf("i = %d, j = %d, H = %d\n", i, j, current->H);

				if(current->H > blockMax[threadIdx.x].score) {
					blockMax[threadIdx.x].score = current->H;
					blockMax[threadIdx.x].row = i;
					blockMax[threadIdx.x].column = j;
				}
			}
		}

		j++;

		__syncthreads();
	}

	int nearestPowof2 = 1;
	while (nearestPowof2 < THREADS_PER_BLOCK)
		nearestPowof2 <<= 1;

	int index = nearestPowof2 / 2;
	while(index != 0) {
		if(threadIdx.x < index && threadIdx.x + index < THREADS_PER_BLOCK) {
			if(blockMax[threadIdx.x].score < blockMax[threadIdx.x + index].score)
				blockMax[threadIdx.x] = blockMax[threadIdx.x + index];
		}

		__syncthreads();
		index /= 2;
	}

	if(threadIdx.x == 0 && blockScore[blockIdx.x].score < blockMax[0].score) {
		blockScore[blockIdx.x] = blockMax[0];
	}
}

void exitWithMsg(char *msg, int exitCode) {
	printf("ERROR\n");
	printf("%s\n\n", msg);
	exit(exitCode);
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
    if(!first.load() || !second.load())
    	exitWithMsg("An error has occured while loading input sequences!", -1);
    else
    	printf("DONE\n\n");
	
    printf("First sequence of length %d:\n%s\n\n", first.getLength(), first.getSequenceName());
    printf("Second sequence of length %d:\n%s\n\n", second.getLength(), second.getSequenceName());

    scoring values;
    printf("Initializing scoring values... ");
    values.match = atoi(argv[3]);
    values.mismatch = atoi(argv[4]);
    values.first = atoi(argv[5]);
    values.extension = atoi(argv[6]);

	if(values.match < 1 || values.mismatch > -1 || values.first > -1 || values.extension > -1)
		exitWithMsg("One or more scoring values were not usable!", -1);
	else {
		printf("DONE\n\nScoring values:\n");
		printf("	>Match: %d\n", values.match);
		printf("	>Mismatch: %d\n", values.mismatch);
		printf("	>First gap: %d\n", values.first);
		printf("	>Gap extension: %d\n\n", values.extension);
	}

    printf("Starting alignment process... ");

    cudaError_t errAlloc, errCpy, errMemset;

    element *matrix;
    int matrixSize = sizeof(element) * (first.getLength() + 1) * (second.getLength() + 1);
    errAlloc = cudaMalloc(&matrix, matrixSize);
    errMemset = cudaMemset(matrix, 0, matrixSize);
    if(errAlloc != 0 || errMemset != 0)
    	exitWithMsg("An error has occured while creating alignment matrix on device.", -1);

    alignmentScore max;
    max.score = 0;
    max.row = 0;
    max.column = 0;

    alignmentScore *blockScores;
    int blockScoreSize = sizeof(alignmentScore) * BLOCKS_PER_GRID;
    blockScores = (alignmentScore *)malloc(blockScoreSize);
    if(blockScores == NULL)
    	exitWithMsg("An error has occured while allocating blockScores array on host.", -1);
    
	alignmentScore *devBlockScore;
    errAlloc = cudaMalloc(&devBlockScore, blockScoreSize);
    errMemset = cudaMemset(devBlockScore, 0, blockScoreSize);
    if(errAlloc != 0 || errMemset != 0)
    	exitWithMsg("An error has occured while creating blockScore array on device.", -1);

    char *devFirst;
    errAlloc = cudaMalloc(&devFirst, first.getLength() * sizeof(char));
    errCpy = cudaMemcpy(devFirst, first.getSequence(),
    		first.getLength() * sizeof(char), cudaMemcpyHostToDevice);
    if(errAlloc != 0 || errCpy != 0)
    	exitWithMsg("An error has occured while transfering first sequence to device.", -1);

    char *devSecond;
    errAlloc = cudaMalloc(&devSecond, second.getLength() * sizeof(char));
    errCpy = cudaMemcpy(devSecond, second.getSequence(),
    		second.getLength() * sizeof(char), cudaMemcpyHostToDevice);
    if(errAlloc != 0 || errCpy != 0)
    	exitWithMsg("An error has occured while transfering second sequence to device.", -1);

	int D = BLOCKS_PER_GRID + first.getLength() / (ALPHA * THREADS_PER_BLOCK) - 1;
	
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    //D + BLOCKS_PER_GRID
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

        if(dk < D + BLOCKS_PER_GRID - 1) {
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
        }
    }
    
    cudaMemcpy(blockScores, devBlockScore, blockScoreSize, cudaMemcpyDeviceToHost);
	for(int i = 0; i < BLOCKS_PER_GRID; i++) {
		if(max.score < blockScores[i].score) {
			max.score = blockScores[i].score;
			max.column = blockScores[i].column;
			max.row = blockScores[i].row;
		}
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    printf("DONE\n\n");


    element *hostMatrix = (element *)malloc(matrixSize);
    cudaMemcpy(hostMatrix, matrix, matrixSize, cudaMemcpyDeviceToHost);
    for(int i = 0; i <= first.getLength(); i++) {
    	for(int j = 0; j <= second.getLength(); j++) {
    		printf("%d ", hostMatrix[j + i * (second.getLength() + 1)].H);
    	}
    	printf("\n");
    }
    free(hostMatrix);


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
