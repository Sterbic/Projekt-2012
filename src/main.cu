#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <algorithm>
#include <vector>

#include "Defines.h"
#include "FASTA.h"
#include "SWutils.h"
#include "Builders.h"
#include "FindAlignment.cuh"
#include "Traceback.cuh"

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
    safeAPIcall(cudaSetDevice(bestGpu.cardNumber), __LINE__);
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

	safeAPIcall(cudaFuncSetCacheConfig(shortPhase, cudaFuncCachePreferShared), __LINE__);
	safeAPIcall(cudaFuncSetCacheConfig(longPhase, cudaFuncCachePreferShared), __LINE__);
	
	safeAPIcall(cudaBindTexture(
			NULL,
			texSecond,
			query.getDevSecond(),
			query.getSecond()->getPaddedLength()
			), __LINE__);

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

		rowBuilder.dumpLong(buffer.hBuffer.up, dk);
    }
    
    kernelTimer.stop();

    safeAPIcall(cudaMemcpy(score, devScore, scoreSize, cudaMemcpyDeviceToHost), __LINE__);
	alignmentScore max = getMaxScore(score, config.blocks * config.threads);

	timer.stop();

    printf("DONE\n\n");

    double gcups = first.getLength() / 1e6 * second.getLength() / (timer.getElapsedTimeMillis());
    printf("\t>Kernel executed in %f s\n", kernelTimer.getElapsedTimeMillis() / 1000);
    printf("\t>Application executed in %f s\n", timer.getElapsedTimeMillis() / 1000);
    printf("\t>Cell updates per second: %lf GCUPS\n", gcups);
    printf("\t>Alignment score: %d at [%d, %d]\n\n", max.score, max.row + 1, max.column + 1);

    safeAPIcall(cudaUnbindTexture(texSecond), __LINE__);

    safeAPIcall(cudaFree(devScore), __LINE__);

    freeGlobalBuffer(&buffer);

    free(score);

    HorizontalBuffer hBuffer;
    hBuffer.up = (int2 *) cudaGetSpaceAndSet(sizeof(int2) * rowBuilder.getRowHeight(), -max.score);
    VerticalBuffer vBuffer;
    initVerticalBuffer(&vBuffer, config);

    int widthOffset = 0;
    int heigthOffset = 0;
    int specialRowIndex = (max.row / rowBuilder.getRowHeight()) * rowBuilder.getRowHeight();
    int chunkSize = rowBuilder.getRowHeight();
    char fileName[50];

    LaunchConfig traceback = getLaunchConfig(chunkSize, bestGpu);
    int paddedChunkHeight = chunkSize;
    if(chunkSize % traceback.blocks != 0)
    	paddedChunkHeight += traceback.blocks - (chunkSize % traceback.blocks);

    int paddedChunkWidth = paddedChunkHeight;
    if(paddedChunkWidth % 4 != 0)
    	paddedChunkWidth += 4 - (paddedChunkHeight % 4);

    char *devRow = (char *) cudaGetSpaceAndSet(paddedChunkWidth * sizeof(char), 0);
    char *devColumn = (char *) cudaGetSpaceAndSet(paddedChunkHeight * sizeof(char), 0);

    int2 *vBusOut = (int2 *) malloc(chunkSize * sizeof(int2));
    int2 *devVBusOut = (int2 *) cudaGetSpaceAndSet(paddedChunkWidth * sizeof(int2), 0);
    char pad[240];
    memset(pad, STAGE_2_PADDING, 240);

    char *firstReversed = query.getFirst()->getReversedSequence(max.row);
    char *secondReversed = query.getSecond()->getReversedSequence(max.column);

    bool gap = false;

    std::vector<TracebackScore> crosspoints;

    TracebackScore maxTrace;
    maxTrace.score = max.score;
    maxTrace.column = max.column;
    maxTrace.row = max.row;
    maxTrace.gap = gap;

    crosspoints.push_back(maxTrace);

	int2 *specialRow = (int2 *) malloc(query.getSecond()->getPaddedLength() * sizeof(int2));
	if(specialRow == NULL)
		exitWithMsg("Error allocating special row.", -1);

    D = traceback.blocks + ceil(((double) std::max(paddedChunkHeight, paddedChunkWidth))
    			/ (ALPHA * traceback.threads)) - 1;

    while(maxTrace.score >= rowBuilder.getRowHeight() * values.match) {

		memset(fileName, 0, 50);
		sprintf(fileName, "temp/row_%d", specialRowIndex);
		printf("%s %d\n", fileName, rowBuilder.getRowHeight());
		FILE *f = fopen(fileName, "rb");
		if(f == NULL)
			exitWithMsg("Error opening special row file.", -1);

		fread(specialRow, sizeof(int2), query.getSecond()->getPaddedLength(), f);
		fclose(f);

		int getVertical = min(chunkSize, maxTrace.row - specialRowIndex);
		safeAPIcall(cudaMemcpy(devColumn, firstReversed + heigthOffset,
				getVertical, cudaMemcpyHostToDevice), __LINE__);

		printf("Padded H = %d\n", paddedChunkHeight);
		for(int i = getVertical; i < paddedChunkHeight - getVertical; i += 240) {
			printf("i = %d ", i);
			safeAPIcall(cudaMemcpy(devColumn + i, pad, min(paddedChunkHeight - i, 240), cudaMemcpyHostToDevice), __LINE__);
		}

		while(widthOffset < max.column) {

			int getNum = min(chunkSize, max.column - widthOffset);
			safeAPIcall(cudaMemcpy(devRow, secondReversed + widthOffset,
					getNum, cudaMemcpyHostToDevice), __LINE__);

			for(int i = getNum; i < paddedChunkWidth - getNum; i += 240)
				safeAPIcall(cudaMemcpy(devRow + i, pad,
						min(paddedChunkWidth - i, 240), cudaMemcpyHostToDevice), __LINE__);

			for(int dk = 0; dk < D + traceback.blocks; dk++) {
				tracebackShort<<<traceback.blocks, traceback.threads, traceback.sharedMemSize>>>(
							dk,
							hBuffer,
							vBuffer,
							devRow,
							paddedChunkWidth,
							devColumn,
							paddedChunkHeight,
							values,
							devVBusOut,
							gap
							);

				tracebackLong<<<traceback.blocks, traceback.threads, traceback.sharedMemSize>>>(
							dk,
							hBuffer,
							vBuffer,
							devRow,
							paddedChunkWidth,
							devColumn,
							paddedChunkHeight,
							values,
							devVBusOut,
							gap
							);
			}

			safeAPIcall(cudaMemcpy(vBusOut, devVBusOut + paddedChunkWidth - chunkSize,
					chunkSize, cudaMemcpyDeviceToHost), __LINE__);



			TracebackScore tracebackScore = getTracebackScore(
					values, gap, specialRowIndex, chunkSize, chunkSize, vBusOut, specialRow);

			maxTrace.score -= tracebackScore.score;
			maxTrace.column -= tracebackScore.column;
			maxTrace.row -= tracebackScore.row;
			maxTrace.gap = tracebackScore.gap;
			gap = tracebackScore.gap;

			printf("Crosspoint [%d, %d] = %d", maxTrace.row, maxTrace.column, maxTrace.score);
			crosspoints.push_back(maxTrace);

			specialRowIndex -= rowBuilder.getRowHeight();
			widthOffset += chunkSize;
			heigthOffset += getVertical;
		}
    }

    if(maxTrace.score != 0) {
    	safeAPIcall(cudaMemcpy(devColumn, firstReversed + heigthOffset,
    			chunkSize, cudaMemcpyHostToDevice), __LINE__);

    	for(int i = chunkSize; i < paddedChunkHeight - chunkSize; i += 240)
			safeAPIcall(cudaMemcpy(devColumn + i, pad,
					min(paddedChunkWidth - i, 240), cudaMemcpyHostToDevice), __LINE__);

    	while(widthOffset < max.column) {
    		int getNum = min(chunkSize, max.column - widthOffset);
			safeAPIcall(cudaMemcpy(devRow, secondReversed + widthOffset,
					getNum, cudaMemcpyHostToDevice), __LINE__);

			for(int i = getNum; i < paddedChunkWidth - getNum; i += 240)
				safeAPIcall(cudaMemcpy(devRow + i, pad,
						min(paddedChunkWidth - i, 240), cudaMemcpyHostToDevice), __LINE__);

			TracebackScore last;
			last.score = maxTrace.score;
			last.column = -10;
			last.row = -10;
			last.gap = false;

			TracebackScore *devLast = (TracebackScore *) cudaGetDeviceCopy(&last, sizeof(TracebackScore));

			for(int dk = 0; dk < D + traceback.blocks; dk++) {
				tracebackLastShort<<<traceback.blocks, traceback.threads, traceback.sharedMemSize>>>(
							dk,
							hBuffer,
							vBuffer,
							devRow,
							paddedChunkWidth,
							devColumn,
							paddedChunkHeight,
							values,
							gap,
							devLast
							);

				safeAPIcall(cudaMemcpy(&last, devLast, sizeof(TracebackScore), cudaMemcpyDeviceToHost), __LINE__);
				if(last.row != -10) break;

				tracebackLastLong<<<traceback.blocks, traceback.threads, traceback.sharedMemSize>>>(
							dk,
							hBuffer,
							vBuffer,
							devRow,
							paddedChunkWidth,
							devColumn,
							paddedChunkHeight,
							values,
							gap,
							devLast
							);

				safeAPIcall(cudaMemcpy(&last, devLast, sizeof(TracebackScore), cudaMemcpyDeviceToHost), __LINE__);
				if(last.row != -10) break;
			}

			crosspoints.push_back(last);
    	}
    }

    safeAPIcall(cudaFree(devColumn), __LINE__);
    safeAPIcall(cudaFree(devRow), __LINE__);
    safeAPIcall(cudaFree(devVBusOut), __LINE__);

    free(specialRow);

    safeAPIcall(cudaFree(hBuffer.up), __LINE__);
    freeVerticalBuffer(&vBuffer);

    return 0;
}
