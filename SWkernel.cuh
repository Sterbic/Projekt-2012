/*
 * SWkernel.cuh
 *
 *  Created on: Dec 4, 2012
 *      Author: Luka
 */

#ifndef SWKERNEL_CUH_
#define SWKERNEL_CUH_

#include "Defines.h"

__device__ int getColumn(int secondLength) {
	return secondLength / gridDim.x * (gridDim.x - blockIdx.x - 1) - threadIdx.x;
}

__device__ int getRow(int dk) {
	return (dk + blockIdx.x - gridDim.x + 1) *
			blockDim.x * ALPHA + threadIdx.x * ALPHA;
}

__device__ void getBlockMax(alignmentScore *blockMax, alignmentScore *blockScore) {
	int nearestPowof2 = 1;
	while (nearestPowof2 < blockDim.x)
		nearestPowof2 <<= 1;

	int index = nearestPowof2 / 2;
	while(index != 0) {
		if(threadIdx.x < index && threadIdx.x + index < blockDim.x) {
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

#endif /* SWKERNEL_CUH_ */
