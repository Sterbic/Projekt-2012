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

__global__ void shortPhase(
		int dk,
		HorizontalBuffer hbuffer,
		VerticalBuffer vbuffer,
		char *first,
		int firstLength,
		char *second,
		int secondLength,
		scoring values,
		alignmentScore *score
		) {

	extern __shared__ int2 iHbuffer[];

	int i = getRow(dk);
	int j = getColumn(secondLength);

	if(j < 0) {
		i -= ALPHA * gridDim.x * blockDim.x;
		j += secondLength;
	}

//	printf("blockIdx.x = %d, threadIdx.x = %d, i = %d, j = %d\n", blockIdx.x, threadIdx.x, i, j);

	char rowBuffer[ALPHA];
	getRowBuffer(i, first, rowBuffer);

	K iBuffer;
	initK(&iBuffer, i, j, &hbuffer, &vbuffer);

	__syncthreads();

	for(int innerDiagonal = 0; innerDiagonal < blockDim.x; innerDiagonal++) {

		if(i >= 0 && i < firstLength) {
			int matchMismatch = values.mismatch;
			if(rowBuffer[0] == second[j])
				matchMismatch = values.match;

			iBuffer.curr0.y = max(iBuffer.left0.y + values.extension, iBuffer.left0.x + values.first);
			iBuffer.curr0.z = max(iBuffer.up.y + values.extension, iBuffer.up.x + values.first);
			iBuffer.curr0.x = max(max(0, iBuffer.curr0.y), max(iBuffer.curr0.z, iBuffer.diagonal + matchMismatch));

		//	current->E = max(left->E + values.extension, left->H + values.first);
		//	current->F = max(up->F + values.extension, up->H + values.first);
		//	current->H = max(max(0, current->E), max(current->F, diagonal->H + matchMissmatch));

		/*	if(blockIdx.x == 2 && threadIdx.x == 0)
				printf("Short [B%d, T%d][%d, %d] = [%d, %d, %d] m=%d first=%c secind=%c\n\n",
						blockIdx.x, threadIdx.x, i + a, j, current->H, current->E, current->F, matchMissmatch, first[i], second[j]); */

			int scoreIndex = threadIdx.x + blockIdx.x * blockDim.x;
			if(iBuffer.curr0.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr0.x;
				score[scoreIndex].row = i;
				score[scoreIndex].column = j;
			}

			iBuffer.curr1.y = max(iBuffer.left1.y + values.extension, iBuffer.left1.x + values.first);
			iBuffer.curr1.z = max(iBuffer.curr0.z + values.extension, iBuffer.curr0.x + values.first);
			iBuffer.curr1.x = max(max(0, iBuffer.curr1.y), max(iBuffer.curr1.z, iBuffer.left0.x + matchMismatch));

			if(iBuffer.curr1.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr1.x;
				score[scoreIndex].row = i + 1;
				score[scoreIndex].column = j;
			}

			iBuffer.curr2.y = max(iBuffer.left2.y + values.extension, iBuffer.left2.x + values.first);
			iBuffer.curr2.z = max(iBuffer.curr1.z + values.extension, iBuffer.curr1.x + values.first);
			iBuffer.curr2.x = max(max(0, iBuffer.curr2.y), max(iBuffer.curr2.z, iBuffer.left1.x + matchMismatch));

			if(iBuffer.curr2.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr2.x;
				score[scoreIndex].row = i + 2;
				score[scoreIndex].column = j;
			}

			iBuffer.curr3.y = max(iBuffer.left3.y + values.extension, iBuffer.left3.x + values.first);
			iBuffer.curr3.z = max(iBuffer.curr2.z + values.extension, iBuffer.curr2.x + values.first);
			iBuffer.curr3.x = max(max(0, iBuffer.curr3.y), max(iBuffer.curr3.z, iBuffer.left2.x + matchMismatch));

			if(iBuffer.curr3.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr3.x;
				score[scoreIndex].row = i + 3;
				score[scoreIndex].column = j;
			}
		}

		__syncthreads();

		if(threadIdx.x < blockDim.x - 1) {
			iHbuffer[threadIdx.x].x = iBuffer.curr3.x;
			iHbuffer[threadIdx.x].y = iBuffer.curr3.z;
		}
		else {
			hbuffer.up[j].x = iBuffer.curr3.x;
			hbuffer.up[j].y = iBuffer.curr3.z;
		}

		j++;

		__syncthreads();

		if(j == secondLength) {
			j = 0;
			i += gridDim.x * ALPHA * blockDim.x;
			getRowBuffer(i, first, rowBuffer);
			initK(&iBuffer, i, j, &hbuffer, &vbuffer);
		}
		else {
			int2 newUp = iHbuffer[threadIdx.x];
			if(threadIdx.x == 0)
				newUp = hbuffer.up[j];

			pushForwardK(&iBuffer, newUp);
		}

		__syncthreads();
	}

	updateVerticalBuffer(&iBuffer, &vbuffer, i);
}

__global__ void longPhase(
		int dk,
		HorizontalBuffer hbuffer,
		VerticalBuffer vbuffer,
		char *first,
		int firstLength,
		char *second,
		int secondLength,
		scoring values,
		alignmentScore *score
		) {

	extern __shared__ int2 iHbuffer[];

	int C = secondLength / gridDim.x;

	int i = getRow(dk);
	int j = getColumn(secondLength) + blockDim.x;

	//printf("blockIdx.x = %d, threadIdx.x = %d, i = %d, j = %d\n", blockIdx.x, threadIdx.x, i, j);

	char rowBuffer[ALPHA];
	getRowBuffer(i, first, rowBuffer);

	K iBuffer;
	initK(&iBuffer, i, j, &hbuffer, &vbuffer);

	__syncthreads();

	for(int innerDiagonal = blockDim.x; innerDiagonal < C; innerDiagonal++) {
		if(i >= 0 && i < firstLength) {
			int matchMismatch = values.mismatch;
			if(rowBuffer[0] == second[j])
				matchMismatch = values.match;

			iBuffer.curr0.y = max(iBuffer.left0.y + values.extension, iBuffer.left0.x + values.first);
			iBuffer.curr0.z = max(iBuffer.up.y + values.extension, iBuffer.up.x + values.first);
			iBuffer.curr0.x = max(max(0, iBuffer.curr0.y), max(iBuffer.curr0.z, iBuffer.diagonal + matchMismatch));

		//	current->E = max(left->E + values.extension, left->H + values.first);
		//	current->F = max(up->F + values.extension, up->H + values.first);
		//	current->H = max(max(0, current->E), max(current->F, diagonal->H + matchMissmatch));

		/*	if(blockIdx.x == 2 && threadIdx.x == 0)
				printf("Short [B%d, T%d][%d, %d] = [%d, %d, %d] m=%d first=%c secind=%c\n\n",
						blockIdx.x, threadIdx.x, i + a, j, current->H, current->E, current->F, matchMissmatch, first[i], second[j]); */

			int scoreIndex = threadIdx.x + blockIdx.x * blockDim.x;
			if(iBuffer.curr0.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr0.x;
				score[scoreIndex].row = i;
				score[scoreIndex].column = j;
			}

			iBuffer.curr1.y = max(iBuffer.left1.y + values.extension, iBuffer.left1.x + values.first);
			iBuffer.curr1.z = max(iBuffer.curr0.z + values.extension, iBuffer.curr0.x + values.first);
			iBuffer.curr1.x = max(max(0, iBuffer.curr1.y), max(iBuffer.curr1.z, iBuffer.left0.x + matchMismatch));

			if(iBuffer.curr1.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr1.x;
				score[scoreIndex].row = i + 1;
				score[scoreIndex].column = j;
			}

			iBuffer.curr2.y = max(iBuffer.left2.y + values.extension, iBuffer.left2.x + values.first);
			iBuffer.curr2.z = max(iBuffer.curr1.z + values.extension, iBuffer.curr1.x + values.first);
			iBuffer.curr2.x = max(max(0, iBuffer.curr2.y), max(iBuffer.curr2.z, iBuffer.left1.x + matchMismatch));

			if(iBuffer.curr2.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr2.x;
				score[scoreIndex].row = i + 2;
				score[scoreIndex].column = j;
			}

			iBuffer.curr3.y = max(iBuffer.left3.y + values.extension, iBuffer.left3.x + values.first);
			iBuffer.curr3.z = max(iBuffer.curr2.z + values.extension, iBuffer.curr2.x + values.first);
			iBuffer.curr3.x = max(max(0, iBuffer.curr3.y), max(iBuffer.curr3.z, iBuffer.left2.x + matchMismatch));

			if(iBuffer.curr3.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr3.x;
				score[scoreIndex].row = i + 3;
				score[scoreIndex].column = j;
			}
		}

		__syncthreads();

		if(threadIdx.x < blockDim.x - 1) {
			iHbuffer[threadIdx.x].x = iBuffer.curr3.x;
			iHbuffer[threadIdx.x].y = iBuffer.curr3.z;
		}
		else {
			hbuffer.up[j].x = iBuffer.curr3.x;
			hbuffer.up[j].y = iBuffer.curr3.z;
		}

		j++;

		__syncthreads();

		int2 newUp = iHbuffer[threadIdx.x];
		if(threadIdx.x == 0)
			newUp = hbuffer.up[j];

		pushForwardK(&iBuffer, newUp);

		__syncthreads();
	}

	updateVerticalBuffer(&iBuffer, &vbuffer, i);
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

    alignmentScore max;
    max.score = -1;
    max.row = -1;
    max.column = -1;

    alignmentScore *score;
    int scoreSize = sizeof(alignmentScore) * config.blocks * config.threads;
    score = (alignmentScore *) malloc(scoreSize);
    if(score == NULL)
    	exitWithMsg("An error has occured while allocating blockScores array on host.", -1);
    
	alignmentScore *devScore = (alignmentScore *) cudaGetSpaceAndSet(scoreSize, 0);

    char *devFirst = (char *) cudaGetDeviceCopy(
    		pFirst->getSequence(),
    		pFirst->getLength() * sizeof(char)
    		);

    char *devSecond = (char *) cudaGetDeviceCopy(
    		pSecond->getSequence(),
    		pSecond->getLength() * sizeof(char)
    		);

    HorizontalBuffer hbuffer;
    hbuffer.up = (int2 *) cudaGetSpaceAndSet((pSecond->getLength() + 1) * sizeof(int2) , 0);

    VerticalBuffer vbuffer;
    vbuffer.diagonal = (int *) cudaGetSpaceAndSet(config.blocks * config.threads * sizeof(int), 0);
    vbuffer.left0 = (int2 *) cudaGetSpaceAndSet(config.blocks * config.threads * sizeof(int2), 0);
    vbuffer.left1 = (int2 *) cudaGetSpaceAndSet(config.blocks * config.threads * sizeof(int2), 0);
    vbuffer.left2 = (int2 *) cudaGetSpaceAndSet(config.blocks * config.threads * sizeof(int2), 0);
    vbuffer.left3 = (int2 *) cudaGetSpaceAndSet(config.blocks * config.threads * sizeof(int2), 0);

	int D = config.blocks + ceil(((double) pFirst->getLength()) / (ALPHA * config.threads)) - 1;

	safeAPIcall(cudaFuncSetCacheConfig(shortPhase, cudaFuncCachePreferShared));
	safeAPIcall(cudaFuncSetCacheConfig(longPhase, cudaFuncCachePreferShared));
	
    cudaTimer timer;
    timer.start();

    for(int dk = 0; dk < D + config.blocks; dk++) {
    	shortPhase<<<config.blocks, config.threads, config.sharedMemSize>>>(
    			dk,
    			hbuffer,
    			vbuffer,
    			devFirst,
    			pFirst->getLength(),
    			devSecond,
    			pSecond->getLength(),
    			values,
    			devScore
    			);

    	safeAPIcall(cudaDeviceSynchronize());

		longPhase<<<config.blocks, config.threads, config.sharedMemSize>>>(
				dk,
    			hbuffer,
    			vbuffer,
				devFirst,
				pFirst->getLength(),
				devSecond,
				pSecond->getLength(),
				values,
				devScore
		);

		safeAPIcall(cudaDeviceSynchronize());
    }
    
    timer.stop();

    safeAPIcall(cudaMemcpy(score, devScore, scoreSize, cudaMemcpyDeviceToHost));
	for(int i = 0; i < config.blocks * config.threads; i++) {
		if(max.score < score[i].score) {
			max.score = score[i].score;
			max.column = score[i].column;
			max.row = score[i].row;
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

    safeAPIcall(cudaFree(devScore));
    safeAPIcall(cudaFree(devFirst));
    safeAPIcall(cudaFree(devSecond));
    safeAPIcall(cudaFree(hbuffer.up));
    safeAPIcall(cudaFree(vbuffer.diagonal));
    safeAPIcall(cudaFree(vbuffer.left0));
    safeAPIcall(cudaFree(vbuffer.left1));
    safeAPIcall(cudaFree(vbuffer.left2));
    safeAPIcall(cudaFree(vbuffer.left3));

    free(score);

    return 0;
}
