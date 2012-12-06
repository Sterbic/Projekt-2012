/*
 * SWkernel.cuh
 *
 *  Created on: Dec 4, 2012
 *      Author: Luka
 */

#ifndef SWKERNEL_CUH_
#define SWKERNEL_CUH_

#include <cuda.h>

#include "Defines.h"

typedef struct {
	int diagonal;
	int2 up;
	int2 left0;
	int2 left1;
	int2 left2;
	int2 left3;
	int3 curr0;
	int3 curr1;
	int3 curr2;
	int3 curr3;
} K;

__device__ void initK(K *k, int i, int j, HorizontalBuffer *hbuffer, VerticalBuffer *vbuffer) {
	int index = (i / ALPHA) % (blockDim.x * gridDim.x);
	k->diagonal = vbuffer->diagonal[index];
	k->left0 = vbuffer->left0[index];
	k->left1 = vbuffer->left1[index];
	k->left2 = vbuffer->left2[index];
	k->left3 = vbuffer->left3[index];

	k->up = hbuffer->up[j];
}

__device__ void pushForwardK(K *k, int2 newUp) {
	k->left0.x = k->curr0.x;
	k->left1.x = k->curr1.x;
	k->left2.x = k->curr2.x;
	k->left3.x = k->curr3.x;

	k->left0.y = k->curr0.y;
	k->left1.y = k->curr1.y;
	k->left2.y = k->curr2.y;
	k->left3.y = k->curr3.y;

	k->diagonal = k->up.x;

	k->up = newUp;
}

__device__ void updateVerticalBuffer(K *k, VerticalBuffer *vbuffer, int i) {
	int index = (i / ALPHA) % (blockDim.x * gridDim.x);
	vbuffer->diagonal[index] = k->up.x;

	vbuffer->left0[index].x = k->curr0.x;
	vbuffer->left1[index].x = k->curr1.x;
	vbuffer->left2[index].x = k->curr2.x;
	vbuffer->left3[index].x = k->curr3.x;

	vbuffer->left0[index].y = k->curr0.y;
	vbuffer->left1[index].y = k->curr1.y;
	vbuffer->left2[index].y = k->curr2.y;
	vbuffer->left3[index].y = k->curr3.y;
}

__device__ int getColumn(int secondLength) {
	return secondLength / gridDim.x * (gridDim.x - blockIdx.x - 1) - threadIdx.x;
}

__device__ int getRow(int dk) {
	return (dk + blockIdx.x - gridDim.x + 1) *
			blockDim.x * ALPHA + threadIdx.x * ALPHA;
}

#endif /* SWKERNEL_CUH_ */
