/*
 * Buffers.h
 *
 *  Created on: Dec 3, 2012
 *      Author: Luka
 */

#ifndef BUFFERS_H_
#define BUFFERS_H_

#include "Defines.h"

typedef struct {
	alignmentScore *blockMax;
	char *columnBuffer;
} SharedMem;

typedef struct {

} VerticalBuffer;

typedef struct {

} HorizontalBuffer;

__device__ void initSharedMem(SharedMem *shared, char *dynamicSharedMem) {
	shared->blockMax = (alignmentScore *) dynamicSharedMem;
	shared->columnBuffer = (char *) &dynamicSharedMem[blockDim.x * sizeof(alignmentScore)];
}

__device__ void getRowBuffer(int i, char *sequence, char *seqBuffer) {
	for(int a = 0; a < ALPHA; a++)
		seqBuffer[a] = sequence[i + a];
}

__device__ void initColumnBuffer(int j, char *sequence, int length, char *seqBuffer) {
	int index = blockDim.x - threadIdx.x - 1;
	seqBuffer[index] = sequence[j];
	j = (j + blockDim.x) % length;
	seqBuffer[index + blockDim.x] = sequence[j];
}

__device__ void updateColumnBuffer(int j, char *seqence, int length, char *seqBuffer) {
	int index = blockDim.x - threadIdx.x - 1;
	seqBuffer[index] = seqBuffer[index + blockDim.x];
	j = (j + blockDim.x) % length;
	seqBuffer[index + blockDim.x] = seqence[j];
}


#endif /* BUFFERS_H_ */
