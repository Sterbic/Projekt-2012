#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "Defines.h"
#include "FASTA.h"
#include "SWutils.h"
#include "Buffers.cuh"
#include "SWkernel.cuh"

typedef struct {
	int H;
	int E;
	int F;
} element;

__global__ void shortPhase(int dk, element *matrix, char *first,
	int firstLength, char *second, int secondLength, scoring values,
	alignmentScore *blockScore) {

	extern __shared__ alignmentScore blockMax[];
	blockMax[threadIdx.x].score = -1;

	int i = getRow(dk);
	int j = getColumn(secondLength);

	if(j < 0) {
		i -= ALPHA * gridDim.x * blockDim.x;
		j += secondLength;
	}

//	printf("blockIdx.x = %d, threadIdx.x = %d, i = %d, j = %d\n", blockIdx.x, threadIdx.x, i, j);

	int cols = secondLength + 1;

	__syncthreads();

	for(int innerDiagonal = 0; innerDiagonal < blockDim.x; innerDiagonal++) {
		element *current = &matrix[j + 1 + (i + 1) * cols];
		element *left = current - 1;
		element *up = current - cols;
		element *diagonal = up - 1;

		char rowBuffer[ALPHA];
		getRowBuffer(i, first, rowBuffer);

		for(int a = 0; a < ALPHA; a++) {
			if(i >= 0 && i < firstLength) {
				int matchMissmatch = values.mismatch;
				if(rowBuffer[a] == second[j])
					matchMissmatch = values.match;

				current->E = max(left->E + values.extension, left->H + values.first);
				current->F = max(up->F + values.extension, up->H + values.first);
				current->H = max(max(0, current->E), max(current->F, diagonal->H + matchMissmatch));

			/*	if(blockIdx.x == 2 && threadIdx.x == 0)
					printf("Short [B%d, T%d][%d, %d] = [%d, %d, %d] m=%d first=%c secind=%c\n\n",
							blockIdx.x, threadIdx.x, i + a, j, current->H, current->E, current->F, matchMissmatch, first[i], second[j]); */

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
			i += gridDim.x * ALPHA * blockDim.x;
			getRowBuffer(i, first, rowBuffer);
		}

	}

	getBlockMax(blockMax,blockScore);
}

__global__ void longPhase(int dk, element *matrix,char *first,
		int firstLength, char *second, int secondLength, scoring values,
		alignmentScore *blockScore) {

	extern __shared__ alignmentScore blockMax[];
	blockMax[threadIdx.x].score = -1;

	int C = secondLength / gridDim.x;

	int i = getRow(dk);
	int j = getColumn(secondLength) + blockDim.x;

	int cols = secondLength + 1;

	//printf("blockIdx.x = %d, threadIdx.x = %d, i = %d, j = %d\n", blockIdx.x, threadIdx.x, i, j);

	char rowBuffer[ALPHA];
	getRowBuffer(i, first, rowBuffer);

	for(int innerDiagonal = blockDim.x; innerDiagonal < C; innerDiagonal++) {
		element *current = &matrix[j + 1 + (i + 1) * cols];
		element *left = current - 1;
		element *up = current - cols;
		element *diagonal = up - 1;

		for(int a = 0; a < ALPHA; a++) {
			if(i >= 0 && i < firstLength && j < secondLength && j >= 0) {
				int matchMissmatch = values.mismatch;
				if(rowBuffer[a] == second[j])
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

int main(int argc, char *argv[]) {
    printf("### Welcome to SWalign v%s\n\n", VERSION);

    if (argc != 7) {
        printf("Expected 6 input arguments, not %d!\n\n", argc - 1);
        return -1;
    }

    FASTAsequence first(argv[1]);
    FASTAsequence second(argv[2]);
    FASTAsequence *pFirst = &first;
    FASTAsequence *pSecond = &second;

    printf("> Loading input sequences... ");
    if(!first.load() || !second.load())
    	exitWithMsg("An error has occured while loading input sequences.", -1);
    else
    	printf("DONE\n\n");
	
    printf("First sequence of length %d:\n%s\n\n", first.getLength(), first.getSequenceName());
    printf("Second sequence of length %d:\n%s\n\n", second.getLength(), second.getSequenceName());

    if(second.getLength() > first.getLength()) {
    	FASTAsequence *temp = pFirst;
    	pFirst = pSecond;
    	pSecond = temp;
    }

    printf("> Looking for CUDA capable cards... ");
    CUDAcard bestGpu = findBestDevice();
    safeAPIcall(cudaSetDevice(bestGpu.cardNumber));
    printf("DONE\n\n");
    printf("Found %d CUDA capable GPU(s), picked GPU number %d:\n",
    		bestGpu.cardsInSystem, bestGpu.cardNumber + 1);
    printCardInfo(bestGpu);
    printf("\n");

    printf("> Initializing launch configuration... ");
    LaunchConfig config = getLaunchConfig(pSecond->getLength(), bestGpu);
    printf("DONE\n\n");
    printLaunchConfig(config);
    printf("\n");

    if(!pFirst->doPaddingForRows() || !pSecond->doPaddingForColumns(config.blocks))
    	exitWithMsg("An error has occured while applying padding on input sequences.", -1);

    scoring values = initScoringValues(argv[3], argv[4], argv[5], argv[6]);

    printf("> Starting alignment process... ");

    int matrixSize = sizeof(element) * (first.getLength() + 1) * (second.getLength() + 1);
    element *devMatrix = (element *) cudaGetSpaceAndSet(matrixSize, 0);

    alignmentScore max;
    max.score = -1;
    max.row = -1;
    max.column = -1;

    alignmentScore *blockScore;
    int blockScoreSize = sizeof(alignmentScore) * config.blocks;
    blockScore = (alignmentScore *) malloc(blockScoreSize);
    if(blockScore == NULL)
    	exitWithMsg("An error has occured while allocating blockScores array on host.", -1);
    
	alignmentScore *devBlockScore = (alignmentScore *) cudaGetSpaceAndSet(blockScoreSize, 0);

    char *devFirst = (char *) cudaGetDeviceCopy(
    		pFirst->getSequence(),
    		pFirst->getLength() * sizeof(char)
    		);

    char *devSecond = (char *) cudaGetDeviceCopy(
    		pSecond->getSequence(),
    		pSecond->getLength() * sizeof(char)
    		);

	int D = config.blocks + ceil(((double) pFirst->getLength()) / (ALPHA * config.threads)) - 1;
	
    cudaTimer timer;
    timer.start();

    for(int dk = 0; dk < D + config.blocks; dk++) {
    	shortPhase<<<config.blocks, config.threads, config.sharedMemSize>>>(
    			dk,
    			devMatrix,
    			devFirst,
    			pFirst->getLength(),
    			devSecond,
    			pSecond->getLength(),
    			values,
    			devBlockScore
    			);

    	safeAPIcall(cudaDeviceSynchronize());

		longPhase<<<config.blocks, config.threads, config.sharedMemSize>>>(
				dk,
				devMatrix,
				devFirst,
				pFirst->getLength(),
				devSecond,
				pSecond->getLength(),
				values,
				devBlockScore
		);

		safeAPIcall(cudaDeviceSynchronize());
    }
    
    timer.stop();

    safeAPIcall(cudaMemcpy(blockScore, devBlockScore, blockScoreSize, cudaMemcpyDeviceToHost));
	for(int i = 0; i < config.blocks; i++) {
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

    printf("Kernel executed in %f s\n", timer.getElapsedTimeMillis() / 1000);

    printf("\nAlignment score: %d at [%d, %d]\n", max.score, max.row + 1, max.column + 1);

    safeAPIcall(cudaFree(devMatrix));
    safeAPIcall(cudaFree(devBlockScore));
    safeAPIcall(cudaFree(devFirst));
    safeAPIcall(cudaFree(devSecond));

    free(blockScore);

    return 0;
}
