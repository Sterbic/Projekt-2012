#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "Defines.h"
#include "FASTA.h"
#include "SWutils.h"
#include "SWkernel.cuh"

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

	if(threadIdx.x == 0)
		iHbuffer[0] = hbuffer.up[j];

	char4 rowBuffer;
	getRowBuffer(i, first, &rowBuffer);

	K iBuffer;
	if(i >= 0 && i < firstLength)
		initK(&iBuffer, i, j, &hbuffer, &vbuffer);

/*	if(threadIdx.x == tid && blockIdx.x == bl) {
		printf("blockIdx.x = %d, threadIdx.x = %d, i = %d, j = %d\n", blockIdx.x, threadIdx.x, i, j);
		printK(&iBuffer);
		printBuffers(&hbuffer, &vbuffer, iHbuffer, secondLength);
	}*/

	int scoreIndex = threadIdx.x + blockIdx.x * blockDim.x;

	for(int innerDiagonal = 0; innerDiagonal < blockDim.x; innerDiagonal++) {

		__syncthreads();

		if(i >= 0 && i < firstLength) {
			int matchMismatch = values.mismatch;
			if(rowBuffer.w == second[j])
				matchMismatch = values.match;

			iBuffer.curr0.y = max(iBuffer.left0.y + values.extension, iBuffer.left0.x + values.first);
			iBuffer.curr0.z = max(iBuffer.up.y + values.extension, iBuffer.up.x + values.first);
			iBuffer.curr0.x = max(max(0, iBuffer.curr0.y), max(iBuffer.curr0.z, iBuffer.diagonal + matchMismatch));

			if(iBuffer.curr0.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr0.x;
				score[scoreIndex].row = i;
				score[scoreIndex].column = j;
			}

			matchMismatch = values.mismatch;
			if(rowBuffer.x == second[j])
				matchMismatch = values.match;

			iBuffer.curr1.y = max(iBuffer.left1.y + values.extension, iBuffer.left1.x + values.first);
			iBuffer.curr1.z = max(iBuffer.curr0.z + values.extension, iBuffer.curr0.x + values.first);
			iBuffer.curr1.x = max(max(0, iBuffer.curr1.y), max(iBuffer.curr1.z, iBuffer.left0.x + matchMismatch));

			if(iBuffer.curr1.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr1.x;
				score[scoreIndex].row = i + 1;
				score[scoreIndex].column = j;
			}

			matchMismatch = values.mismatch;
			if(rowBuffer.y == second[j])
				matchMismatch = values.match;

			iBuffer.curr2.y = max(iBuffer.left2.y + values.extension, iBuffer.left2.x + values.first);
			iBuffer.curr2.z = max(iBuffer.curr1.z + values.extension, iBuffer.curr1.x + values.first);
			iBuffer.curr2.x = max(max(0, iBuffer.curr2.y), max(iBuffer.curr2.z, iBuffer.left1.x + matchMismatch));

			if(iBuffer.curr2.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr2.x;
				score[scoreIndex].row = i + 2;
				score[scoreIndex].column = j;
			}

			matchMismatch = values.mismatch;
			if(rowBuffer.z == second[j])
				matchMismatch = values.match;

			iBuffer.curr3.y = max(iBuffer.left3.y + values.extension, iBuffer.left3.x + values.first);
			iBuffer.curr3.z = max(iBuffer.curr2.z + values.extension, iBuffer.curr2.x + values.first);
			iBuffer.curr3.x = max(max(0, iBuffer.curr3.y), max(iBuffer.curr3.z, iBuffer.left2.x + matchMismatch));

			if(iBuffer.curr3.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr3.x;
				score[scoreIndex].row = i + 3;
				score[scoreIndex].column = j;
			}

			if(threadIdx.x < blockDim.x - 1) {
				iHbuffer[threadIdx.x + 1].x = iBuffer.curr3.x;
				iHbuffer[threadIdx.x + 1].y = iBuffer.curr3.z;
			}
			else {
				hbuffer.up[j].x = iBuffer.curr3.x;
				hbuffer.up[j].y = iBuffer.curr3.z;
			}
		}

		j++;

		__syncthreads();

	/*	if(threadIdx.x == tid && blockIdx.x == bl)
			printBuffers(&hbuffer, &vbuffer, iHbuffer, secondLength);*/

		if(j == secondLength) {
			j = 0;
			i += gridDim.x * ALPHA * blockDim.x;
			getRowBuffer(i, first, &rowBuffer);
			initK(&iBuffer, i, j, iHbuffer, &vbuffer);
		}
		else {
			int2 newUp;
			if(threadIdx.x > 0)
				newUp = iHbuffer[threadIdx.x];
			else
				newUp = hbuffer.up[j];

			pushForwardK(&iBuffer, newUp);
		}

/*		if(threadIdx.x == tid && blockIdx.x == bl) {
			printf("blockIdx.x = %d, threadIdx.x = %d, i = %d, j = %d\n", blockIdx.x, threadIdx.x, i, j);
			printK(&iBuffer);
			printBuffers(&hbuffer, &vbuffer, iHbuffer, secondLength);
		}*/

		__syncthreads();
	}

	if (i >= 0 && i < firstLength) {
		updateVerticalBuffer(&iBuffer, &vbuffer, i);
		if(threadIdx.x < blockDim.x -1)
			hbuffer.up[j - 1] = iHbuffer[threadIdx.x + 1];
	}

/*	if(threadIdx.x == tid && blockIdx.x == bl) {
		printBuffers(&hbuffer, &vbuffer, iHbuffer, secondLength);
	}*/
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

	char4 rowBuffer;
	getRowBuffer(i, first, &rowBuffer);

	K iBuffer;
	if(i >= 0 && i < firstLength)
		initK(&iBuffer, i, j, &hbuffer, &vbuffer);

/*	if(threadIdx.x == tid && blockIdx.x == bl) {
		printf("blockIdx.x = %d, threadIdx.x = %d, i = %d, j = %d\n", blockIdx.x, threadIdx.x, i, j);
		printK(&iBuffer);
		printBuffers(&hbuffer, &vbuffer, iHbuffer, secondLength);
	} */

	int scoreIndex = threadIdx.x + blockIdx.x * blockDim.x;

	__syncthreads();

	for(int innerDiagonal = blockDim.x; innerDiagonal < C; innerDiagonal++) {
		if(i >= 0 && i < firstLength) {
			int matchMismatch = values.mismatch;
			if(rowBuffer.w == second[j])
				matchMismatch = values.match;

			iBuffer.curr0.y = max(iBuffer.left0.y + values.extension, iBuffer.left0.x + values.first);
			iBuffer.curr0.z = max(iBuffer.up.y + values.extension, iBuffer.up.x + values.first);
			iBuffer.curr0.x = max(max(0, iBuffer.curr0.y), max(iBuffer.curr0.z, iBuffer.diagonal + matchMismatch));

			if(iBuffer.curr0.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr0.x;
				score[scoreIndex].row = i;
				score[scoreIndex].column = j;
			}

			matchMismatch = values.mismatch;
			if(rowBuffer.x == second[j])
				matchMismatch = values.match;

			iBuffer.curr1.y = max(iBuffer.left1.y + values.extension, iBuffer.left1.x + values.first);
			iBuffer.curr1.z = max(iBuffer.curr0.z + values.extension, iBuffer.curr0.x + values.first);
			iBuffer.curr1.x = max(max(0, iBuffer.curr1.y), max(iBuffer.curr1.z, iBuffer.left0.x + matchMismatch));

			if(iBuffer.curr1.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr1.x;
				score[scoreIndex].row = i + 1;
				score[scoreIndex].column = j;
			}

			matchMismatch = values.mismatch;
			if(rowBuffer.y == second[j])
				matchMismatch = values.match;

			iBuffer.curr2.y = max(iBuffer.left2.y + values.extension, iBuffer.left2.x + values.first);
			iBuffer.curr2.z = max(iBuffer.curr1.z + values.extension, iBuffer.curr1.x + values.first);
			iBuffer.curr2.x = max(max(0, iBuffer.curr2.y), max(iBuffer.curr2.z, iBuffer.left1.x + matchMismatch));

			if(iBuffer.curr2.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr2.x;
				score[scoreIndex].row = i + 2;
				score[scoreIndex].column = j;
			}

			matchMismatch = values.mismatch;
			if(rowBuffer.z == second[j])
				matchMismatch = values.match;

			iBuffer.curr3.y = max(iBuffer.left3.y + values.extension, iBuffer.left3.x + values.first);
			iBuffer.curr3.z = max(iBuffer.curr2.z + values.extension, iBuffer.curr2.x + values.first);
			iBuffer.curr3.x = max(max(0, iBuffer.curr3.y), max(iBuffer.curr3.z, iBuffer.left2.x + matchMismatch));

			if(iBuffer.curr3.x > score[scoreIndex].score) {
				score[scoreIndex].score = iBuffer.curr3.x;
				score[scoreIndex].row = i + 3;
				score[scoreIndex].column = j;
			}

			if(threadIdx.x < blockDim.x - 1) {
				iHbuffer[threadIdx.x + 1].x = iBuffer.curr3.x;
				iHbuffer[threadIdx.x + 1].y = iBuffer.curr3.z;
			}
			else {
				hbuffer.up[j].x = iBuffer.curr3.x;
				hbuffer.up[j].y = iBuffer.curr3.z;
			}
		}

	/*	if(threadIdx.x == tid && blockIdx.x == bl) {
			printf("blockIdx.x = %d, threadIdx.x = %d, i = %d, j = %d\n", blockIdx.x, threadIdx.x, i, j);
			printK(&iBuffer);
		}*/

		j++;

		__syncthreads();

	/*	if(threadIdx.x == tid && blockIdx.x == bl)
			printBuffers(&hbuffer, &vbuffer, iHbuffer, secondLength);*/

		int2 newUp;
		if(threadIdx.x > 0)
			newUp = iHbuffer[threadIdx.x];
		else
			newUp = hbuffer.up[j];

		pushForwardK(&iBuffer, newUp);

	/*	if(threadIdx.x == tid && blockIdx.x == bl) {
			printf("blockIdx.x = %d, threadIdx.x = %d, i = %d, j = %d\n", blockIdx.x, threadIdx.x, i, j);
			printK(&iBuffer);
			printBuffers(&hbuffer, &vbuffer, iHbuffer, secondLength);
		}*/

		__syncthreads();
	}

	if (i >= 0 && i < firstLength) {
		updateVerticalBuffer(&iBuffer, &vbuffer, i);
		if(threadIdx.x < blockDim.x -1)
			hbuffer.up[j - 1] = iHbuffer[threadIdx.x + 1];
	}

/*	if(threadIdx.x == tid && blockIdx.x == bl) {
		printBuffers(&hbuffer, &vbuffer, iHbuffer, secondLength);
	}*/
}

int main(int argc, char *argv[]) {
    printf("### Welcome to SWalign v%s\n\n", VERSION);
    cudaTimer timer;
    timer.start();

    if (argc != 7) {
        printf("Expected 6 input arguments, not %d!\n\n", argc - 1);
        return -1;
    }

    FASTAsequence first(argv[1]);
    FASTAsequence second(argv[2]);

    printf("> Loading input sequences... ");
    if(!first.load() || !second.load())
    	exitWithMsg("An error has occured while loading input sequences.", -1);
    else
    	printf("DONE\n\n");
	
    printf("First sequence of length %d:\n%s\n\n", first.getLength(), first.getSequenceName());
    printf("Second sequence of length %d:\n%s\n\n", second.getLength(), second.getSequenceName());

    printf("> Looking for CUDA capable cards... ");
    CUDAcard bestGpu = findBestDevice();
    safeAPIcall(cudaSetDevice(bestGpu.cardNumber));
    printf("DONE\n\n");
    printf("Found %d CUDA capable GPU(s), picked GPU number %d:\n",
    		bestGpu.cardsInSystem, bestGpu.cardNumber + 1);
    printCardInfo(bestGpu);
    printf("\n");

    printf("> Initializing launch configuration... ");
    LaunchConfig config = getLaunchConfig(
    		min(first.getLength(), second.getLength()),
    		bestGpu);
    printf("DONE\n\n");
    printLaunchConfig(config);

    printf("\n> Preparing SWquerry... ");
    SWquerry querry(&first, &second);
    querry.prepare(config);
    printf("DONE\n\n");

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

    GlobalBuffer buffer;
    initGlobalBuffer(&buffer, querry.getSecond()->getPaddedLength(), config);

	int D = config.blocks + ceil(((double) querry.getFirst()->getPaddedLength())
			/ (ALPHA * config.threads)) - 1;

	safeAPIcall(cudaFuncSetCacheConfig(shortPhase, cudaFuncCachePreferShared));
	safeAPIcall(cudaFuncSetCacheConfig(longPhase, cudaFuncCachePreferShared));
	
    cudaTimer kernelTimer;
    kernelTimer.start();

    for(int dk = 0; dk < D + config.blocks; dk++) {
    	shortPhase<<<config.blocks, config.threads, config.sharedMemSize>>>(
    			dk,
    			buffer.hBuffer,
    			buffer.vBuffer,
    			querry.getDevFirst(),
    			querry.getFirst()->getPaddedLength(),
    			querry.getDevSecond(),
    			querry.getSecond()->getPaddedLength(),
    			values,
    			devScore
    			);

    	safeAPIcall(cudaDeviceSynchronize());
		longPhase<<<config.blocks, config.threads, config.sharedMemSize>>>(
				dk,
    			buffer.hBuffer,
    			buffer.vBuffer,
    			querry.getDevFirst(),
    			querry.getFirst()->getPaddedLength(),
    			querry.getDevSecond(),
    			querry.getSecond()->getPaddedLength(),
				values,
				devScore
				);

		safeAPIcall(cudaDeviceSynchronize());
    }
    
    kernelTimer.stop();

    safeAPIcall(cudaMemcpy(score, devScore, scoreSize, cudaMemcpyDeviceToHost));
	for(int i = 0; i < config.blocks * config.threads; i++) {
		if(max.score < score[i].score) {
			max.score = score[i].score;
			max.column = score[i].column;
			max.row = score[i].row;
		}
    }

	timer.stop();

    printf("DONE\n\n");

    printf("Kernel executed in %f s\n", kernelTimer.getElapsedTimeMillis() / 1000);
    printf("Application executed in %f s\n", timer.getElapsedTimeMillis() / 1000);

    printf("\nAlignment score: %d at [%d, %d]\n", max.score, max.row + 1, max.column + 1);

    safeAPIcall(cudaFree(devScore));

    freeGlobalBuffer(&buffer);

    free(score);

    return 0;
}
