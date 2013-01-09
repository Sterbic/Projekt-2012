#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "Defines.h"
#include "FASTA.h"
#include "SWutils.h"
#include "RowBuilder.h"
#include "FindAlignment.cuh"

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

    printf("\n> Preparing SWquery... ");
    SWquery query(&first, &second);
    query.prepare(config);
    printf("DONE\n\n");

    scoring values = initScoringValues(argv[3], argv[4], argv[5], argv[6]);

    printf("> Starting alignment process... ");

    alignmentScore *score;
    int scoreSize = sizeof(alignmentScore) * config.blocks * config.threads;
    score = (alignmentScore *) malloc(scoreSize);
    if(score == NULL)
    	exitWithMsg("An error has occured while allocating blockScores array on host.", -1);
    
	alignmentScore *devScore = (alignmentScore *) cudaGetSpaceAndSet(scoreSize, 0);

    GlobalBuffer buffer;
    initGlobalBuffer(&buffer, query.getSecond()->getPaddedLength(), config);

	int D = config.blocks + ceil(((double) query.getFirst()->getPaddedLength())
			/ (ALPHA * config.threads)) - 1;

	safeAPIcall(cudaFuncSetCacheConfig(shortPhase, cudaFuncCachePreferShared));
	safeAPIcall(cudaFuncSetCacheConfig(longPhase, cudaFuncCachePreferShared));
	
	safeAPIcall(cudaBindTexture(
			NULL,
			texSecond,
			query.getDevSecond(),
			query.getSecond()->getPaddedLength()
			));

	RowBuilder rowBuilder(
			query.getFirst()->getPaddedLength(),
			query.getSecond()->getPaddedLength(),
			&config
			);

    cudaTimer kernelTimer;
    kernelTimer.start();

    for(int dk = 0; dk < D + config.blocks; dk++) {
    	shortPhase<<<config.blocks, config.threads, config.sharedMemSize>>>(
    			dk,
    			buffer.hBuffer,
    			buffer.vBuffer,
    			query.getDevFirst(),
    			query.getFirst()->getPaddedLength(),
    			query.getSecond()->getPaddedLength(),
    			values,
    			devScore
    			);

    	safeAPIcall(cudaDeviceSynchronize());
    	rowBuilder.dumpShort(buffer.hBuffer.up, dk);

		longPhase<<<config.blocks, config.threads, config.sharedMemSize>>>(
				dk,
    			buffer.hBuffer,
    			buffer.vBuffer,
    			query.getDevFirst(),
    			query.getFirst()->getPaddedLength(),
    			query.getSecond()->getPaddedLength(),
				values,
				devScore
				);

		safeAPIcall(cudaDeviceSynchronize());
		rowBuilder.dumpLong(buffer.hBuffer.up, dk);
    }
    
    kernelTimer.stop();

    safeAPIcall(cudaMemcpy(score, devScore, scoreSize, cudaMemcpyDeviceToHost));
	alignmentScore max = getMaxScore(score, config.blocks * config.threads);

	timer.stop();

    printf("DONE\n\n");

    double gcups = first.getLength() / 1e6 * second.getLength() / (timer.getElapsedTimeMillis());
    printf("\t>Kernel executed in %f s\n", kernelTimer.getElapsedTimeMillis() / 1000);
    printf("\t>Application executed in %f s\n", timer.getElapsedTimeMillis() / 1000);
    printf("\t>Cell updates per second: %lf GCUPS\n", gcups);
    printf("\t>Alignment score: %d at [%d, %d]\n\n", max.score, max.row + 1, max.column + 1);

    safeAPIcall(cudaUnbindTexture(texSecond));

    safeAPIcall(cudaFree(devScore));

    freeGlobalBuffer(&buffer);

    free(score);

    HorizontalBuffer hBuffer;
    hBuffer.up = (int2 *) cudaGetSpaceAndSet(sizeof(int2) * rowBuilder.getRowHeight(), - max.score);
    VerticalBuffer vBuffer;
    initVerticalBuffer(&vBuffer, config);

    int offset = 0;
    int chunkSize = (max.column + 24) / 25;
    chunkSize = (chunkSize + 3) / 4;

    int2 *vBusOut = (int2 *) malloc(chunkSize * sizeof(int2));

    char *firstReversed = query.getFirst()->getReversedSequence();
    char *secondReversed = query.getSecond()->getReversedSequence();

    char *firstChunk;
    char *secondChunk;


    safeAPIcall(cudaFree(hBuffer.up));
    freeVerticalBuffer(&vBuffer);

    return 0;
}
