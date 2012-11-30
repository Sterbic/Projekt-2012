#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "FASTA.h"

#define VERSION "0.2.6"

#define THREADS_PER_BLOCK 32
#define BLOCKS_PER_GRID 16
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

__device__ int getColumn(int secondLength) {
	return secondLength / BLOCKS_PER_GRID * (BLOCKS_PER_GRID - blockIdx.x - 1) - threadIdx.x;
}

__device__ int getRow(int dk) {
	return (dk + blockIdx.x - BLOCKS_PER_GRID + 1) *
			THREADS_PER_BLOCK * ALPHA + threadIdx.x * ALPHA;
}

__device__ void getBlockMax(alignmentScore *blockMax, alignmentScore *blockScore) {
	int nearestPowof2 = 1;
		while (nearestPowof2 < THREADS_PER_BLOCK)
			nearestPowof2 <<= 1;

		int index = nearestPowof2 / 2;
		while(index != 0) {
			if(threadIdx.x < index && threadIdx.x + index < THREADS_PER_BLOCK) {
				if(blockMax[threadIdx.x].score <= blockMax[threadIdx.x + index].score)
					blockMax[threadIdx.x] = blockMax[threadIdx.x + index];
			}

			__syncthreads();
			index /= 2;
		}

		if(threadIdx.x == 0 && blockScore[blockIdx.x].score < blockMax[0].score) {
			blockScore[blockIdx.x] = blockMax[0];
		}
}

__global__ void shortPhase(int dk, element *matrix, char *first,
	int firstLength, char *second, int secondLength, scoring values,
	alignmentScore *blockScore) {

	__shared__ alignmentScore blockMax[THREADS_PER_BLOCK];
	blockMax[threadIdx.x].score = -1;

//	int i = ALPHA * (blockIdx.x * THREADS_PER_BLOCK + threadIdx.x);
//	int j = secondLength / BLOCKS_PER_GRID * (dk - blockIdx.x) - threadIdx.x;

	int i = getRow(dk);
	int j = getColumn(secondLength);

	if(j < 0) {
		i -= ALPHA * BLOCKS_PER_GRID * THREADS_PER_BLOCK;
		j += secondLength;
	}

//	printf("blockIdx.x = %d, threadIdx.x = %d, i = %d, j = %d\n", blockIdx.x, threadIdx.x, i, j);

	int cols = secondLength + 1;

	__syncthreads();

	for(int innerDiagonal = 0; innerDiagonal < THREADS_PER_BLOCK; innerDiagonal++) {
		element *current = &matrix[j + 1 + (i + 1) * cols];
		element *left = current - 1;
		element *up = current - cols;
		element *diagonal = up - 1;

		for(int a = 0; a < ALPHA; a++) {
			if(i >= 0 && i < firstLength) {
				int matchMissmatch = values.mismatch;
				if(first[i + a] == second[j])
					matchMissmatch = values.match;

				current->E = max(left->E + values.extension, left->H + values.first);
				current->F = max(up->F + values.extension, up->H + values.first);
				current->H = max(max(0, current->E), max(current->F, diagonal->H + matchMissmatch));

				if(blockIdx.x == 2 && threadIdx.x == 0)
					printf("Short [B%d, T%d][%d, %d] = [%d, %d, %d] m=%d first=%c secind=%c\n\n",
							blockIdx.x, threadIdx.x, i + a, j, current->H, current->E, current->F, matchMissmatch, first[i], second[j]);

				if(current->H > blockMax[threadIdx.x].score) {
					blockMax[threadIdx.x].score = current->H;
					blockMax[threadIdx.x].row = i + a;
					blockMax[threadIdx.x].column = j;
				}

				up = current;
				diagonal = left;
				current += cols;
				left = current - 1;
			}
		}

		__syncthreads();

		j++;

		if(j == secondLength) {
			j = 0;
			i += BLOCKS_PER_GRID * ALPHA * THREADS_PER_BLOCK;
		}

	}

	getBlockMax(blockMax,blockScore);
}

__global__ void longPhase(int dk, element *matrix,char *first,
		int firstLength, char *second, int secondLength, scoring values,
		alignmentScore *blockScore) {

	__shared__ alignmentScore blockMax[THREADS_PER_BLOCK];
	blockMax[threadIdx.x].score = -1;

	int C = secondLength / BLOCKS_PER_GRID;

//	int i = ALPHA * (blockIdx.x * THREADS_PER_BLOCK + threadIdx.x);
//	int j = C * (dk - blockIdx.x) - threadIdx.x + THREADS_PER_BLOCK;

	int i = getRow(dk);
	int j = getColumn(secondLength) + THREADS_PER_BLOCK;

	int cols = secondLength + 1;

	//printf("blockIdx.x = %d, threadIdx.x = %d, i = %d, j = %d\n", blockIdx.x, threadIdx.x, i, j);

	for(int innerDiagonal = THREADS_PER_BLOCK; innerDiagonal < C; innerDiagonal++) {
		element *current = &matrix[j + 1 + (i + 1) * cols];
		element *left = current - 1;
		element *up = current - cols;
		element *diagonal = up - 1;

		for(int a = 0; a < ALPHA; a++) {
			if(i >= 0 && i < firstLength && j < secondLength && j >= 0) {
				int matchMissmatch = values.mismatch;
				if(first[i + a] == second[j])
				matchMissmatch = values.match;

				current->E = max(left->E + values.extension, left->H + values.first);
				current->F = max(up->F + values.extension, up->H + values.first);
				current->H = max(max(0, current->E), max(current->F, diagonal->H + matchMissmatch));

				//printf("Long [B%d, T%d][%d, %d] = [%d, %d, %d]\n\n", blockIdx.x, threadIdx.x, i + a, j, current->H, current->E, current->F);

				if(current->H > blockMax[threadIdx.x].score) {
					blockMax[threadIdx.x].score = current->H;
					blockMax[threadIdx.x].row = i + a;
					blockMax[threadIdx.x].column = j;
				}

				up = current;
				diagonal = left;
				current += cols;
				left = current - 1;
			}
		}

		j++;

		__syncthreads();
	}

	getBlockMax(blockMax,blockScore);
}

void exitWithMsg(const char *msg, int exitCode) {
	printf("ERROR\n");
	printf("%s\n\n", msg);
	exit(exitCode);
}

scoring initScoringValues(const char *match, const char *mismath,
		const char *first, const char *extension) {

	scoring values;
	printf("Initializing scoring values... ");
	values.match = atoi(match);
	values.mismatch = atoi(mismath);
	values.first = atoi(first);
	values.extension = atoi(extension);

	if(values.match < 1 || values.mismatch > -1 || values.first > -1 || values.extension > -1)
		exitWithMsg("One or more scoring values were not usable!", -1);
	else {
		printf("DONE\n\nScoring values:\n");
		printf("	>Match: %d\n", values.match);
		printf("	>Mismatch: %d\n", values.mismatch);
		printf("	>First gap: %d\n", values.first);
		printf("	>Gap extension: %d\n\n", values.extension);
	}

	return values;
}

int main(int argc, char *argv[]) {
    printf("> Welcome to SWaligner v%s\n\n", VERSION);

    if (argc != 7) {
        printf("Expected 6 input arguments, not %d!\n\n", argc - 1);
        return -1;
    }

    FASTAsequence first(argv[1]);
    FASTAsequence second(argv[2]);
    FASTAsequence *pFirst = &first;
    FASTAsequence *pSecond = &second;

    printf("Loading input sequences... ");
    if(!first.load() || !second.load())
    	exitWithMsg("An error has occured while loading input sequences!", -1);
    else
    	printf("DONE\n\n");
	
    printf("First sequence of length %d:\n%s\n\n", first.getLength(), first.getSequenceName());
    printf("Second sequence of length %d:\n%s\n\n", second.getLength(), second.getSequenceName());

    if(second.getLength() > first.getLength()) {
    	FASTAsequence *temp = pFirst;
    	pFirst = pSecond;
    	pSecond = temp;
    }

    if(!pFirst->doPaddingForRows(ALPHA) || !pSecond->doPaddingForColumns(BLOCKS_PER_GRID))
    	exitWithMsg("An error has occured while applying padding on input sequences.", -1);

    printf("First sequence of length %d:\n%s\n\n", pFirst->getLength(), pFirst->getSequenceName());
    printf("Second sequence of length %d:\n%s\n\n", pSecond->getLength(), pSecond->getSequenceName());

    scoring values = initScoringValues(argv[3], argv[4], argv[5], argv[6]);

    printf("Starting alignment process... ");

    cudaError_t errAlloc, errCpy, errMemset;

    element *devMatrix;
    int matrixSize = sizeof(element) * (first.getLength() + 1) * (second.getLength() + 1);
    errAlloc = cudaMalloc(&devMatrix, matrixSize);
    errMemset = cudaMemset(devMatrix, 0, matrixSize);
    if(errAlloc != 0 || errMemset != 0)
    	exitWithMsg("An error has occured while creating alignment devMatrix on device.", -1);

    alignmentScore max;
    max.score = -1;
    max.row = -1;
    max.column = -1;

    alignmentScore *blockScore;
    int blockScoreSize = sizeof(alignmentScore) * BLOCKS_PER_GRID;
    blockScore = (alignmentScore *)malloc(blockScoreSize);
    if(blockScore == NULL)
    	exitWithMsg("An error has occured while allocating blockScores array on host.", -1);
    
	alignmentScore *devBlockScore;
    errAlloc = cudaMalloc(&devBlockScore, blockScoreSize);
    errMemset = cudaMemset(devBlockScore, 0, blockScoreSize);
    if(errAlloc != 0 || errMemset != 0)
    	exitWithMsg("An error has occured while creating blockScore array on device.", -1);

    char *devFirst;
    errAlloc = cudaMalloc(&devFirst, pFirst->getLength() * sizeof(char));
    errCpy = cudaMemcpy(devFirst, pFirst->getSequence(),
    		pFirst->getLength() * sizeof(char), cudaMemcpyHostToDevice);
    if(errAlloc != 0 || errCpy != 0)
    	exitWithMsg("An error has occured while transfering first sequence to device.", -1);

    char *devSecond;
    errAlloc = cudaMalloc(&devSecond, pSecond->getLength() * sizeof(char));
    errCpy = cudaMemcpy(devSecond, pSecond->getSequence(),
    		pSecond->getLength() * sizeof(char), cudaMemcpyHostToDevice);
    if(errAlloc != 0 || errCpy != 0)
    	exitWithMsg("An error has occured while transfering second sequence to device.", -1);

	int D = BLOCKS_PER_GRID + ceil(((double) pFirst->getLength()) / (ALPHA * THREADS_PER_BLOCK)) - 1;
	
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // D + BLOCKS_PER_GRID
    for(int dk = 0; dk < D + BLOCKS_PER_GRID; dk++) {
    	shortPhase<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    			dk,
    			devMatrix,
    			devFirst,
    			pFirst->getLength(),
    			devSecond,
    			pSecond->getLength(),
    			values,
    			devBlockScore
    			);

        if(cudaDeviceSynchronize() != 0)
        	exitWithMsg(cudaGetErrorString(cudaGetLastError()), -1);

		longPhase<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
				dk,
				devMatrix,
				devFirst,
				pFirst->getLength(),
				devSecond,
				pSecond->getLength(),
				values,
				devBlockScore
		);

		cudaDeviceSynchronize();
        if(cudaGetLastError() != 0)
        	exitWithMsg(cudaGetErrorString(cudaGetLastError()), -1);
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaMemcpy(blockScore, devBlockScore, blockScoreSize, cudaMemcpyDeviceToHost);
	for(int i = 0; i < BLOCKS_PER_GRID; i++) {
		if(max.score < blockScore[i].score) {
			max.score = blockScore[i].score;
			max.column = blockScore[i].column;
			max.row = blockScore[i].row;
		}
    }

    printf("DONE\n\n");
/*
    element *hostMatrix = (element *)malloc(matrixSize);
    cudaMemcpy(hostMatrix, devMatrix, matrixSize, cudaMemcpyDeviceToHost);
    for(int i = 0; i <= pFirst->getLength(); i++) {
    	for(int j = 0; j <= pSecond->getLength(); j++) {
    		printf("%d\t", hostMatrix[j + i * (pSecond->getLength() + 1)].H);
    	}
    	printf("\n");
    }
    free(hostMatrix);
*/
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Kernel executed in %f s\n", time / 1000);

    printf("\nAlignment score: %d at [%d, %d]\n", max.score, max.row, max.column);

    cudaFree(devMatrix);
    cudaFree(devBlockScore);
    cudaFree(devFirst);
    cudaFree(devSecond);

    free(blockScore);

    return 0;
}
