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

__device__ void printK(K *k) {
	printf("\n##########\nB: %d, T: %d\n", blockIdx.x, threadIdx.x);
	printf("|\t%d\t|\t%d %d\t|\n", k->diagonal, k->up.x, k->up.y);
	printf("|\t%d %d\t|\t%d %d %d\t|\n", k->left0.x, k->left0.y, k->curr0.x, k->curr0.y, k->curr0.z);
	printf("|\t%d %d\t|\t%d %d %d\t|\n", k->left1.x, k->left1.y, k->curr1.x, k->curr1.y, k->curr1.z);
	printf("|\t%d %d\t|\t%d %d %d\t|\n", k->left2.x, k->left2.y, k->curr2.x, k->curr2.y, k->curr2.z);
	printf("|\t%d %d\t|\t%d %d %d\t|\n", k->left3.x, k->left3.y, k->curr3.x, k->curr3.y, k->curr3.z);
	printf("##########\n");
}

__device__ void printBuffers(HorizontalBuffer *h, VerticalBuffer *v, int2 *localh, int hlen) {
	printf("Horizontal: { ");
	for(int i = 0; i < hlen; i++)
		printf("[%d, %d] ", h->up[i].x, h->up[i].y);
	printf("}\n\n");

	printf("Local h: { ");
		for(int i = 0; i < blockDim.x; i++)
			printf("[%d, %d] ", localh[i].x, localh[i].y);
	printf("}\n\n");

	printf("Vertical:\n");
	for(int i = 0; i < blockDim.x * gridDim.x; i++) {
		printf("V[%d] = { D[%d], L0[%d, %d], L1[%d, %d], L2[%d, %d], L3[%d, %d] }\n", i, v->diagonal[i],
				v->left0[i].x, v->left0[i].y, v->left1[i].x, v->left1[i].y,
				v->left2[i].x, v->left2[i].y, v->left3[i].x, v->left3[i].y);
	}
	printf("\n\n");
}

__device__ void initK(K *k, int i, int j, HorizontalBuffer *hbuffer, VerticalBuffer *vbuffer) {
	int index = (i / ALPHA) % (blockDim.x * gridDim.x);
	k->diagonal = vbuffer->diagonal[index];
	k->left0 = vbuffer->left0[index];
	k->left1 = vbuffer->left1[index];
	k->left2 = vbuffer->left2[index];
	k->left3 = vbuffer->left3[index];

	k->up = hbuffer->up[j];
}

__device__ void getK0(K *k) {
	k->diagonal = 0;

	int2 zero;
	zero.x = 0;
	zero.y = 0;
	k->left0 = zero;
	k->left1 = zero;
	k->left2 = zero;
	k->left3 = zero;
}

__device__ void initK(K *k, int i, int j, int2 * lHbuffer, VerticalBuffer *vbuffer) {
	int index = (i / ALPHA) % (blockDim.x * gridDim.x);
	k->diagonal = vbuffer->diagonal[index];
	k->left0 = vbuffer->left0[index];
	k->left1 = vbuffer->left1[index];
	k->left2 = vbuffer->left2[index];
	k->left3 = vbuffer->left3[index];

	k->up = lHbuffer[threadIdx.x];
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
	vbuffer->diagonal[index] = k->diagonal;

	vbuffer->left0[index] = k->left0;
	vbuffer->left1[index] = k->left1;
	vbuffer->left2[index] = k->left2;
	vbuffer->left3[index] = k->left3;
}

__device__ int getColumn(int secondLength) {
	return secondLength / gridDim.x * (gridDim.x - blockIdx.x - 1) - threadIdx.x;
}

__device__ int getRow(int dk) {
	return (dk + blockIdx.x - gridDim.x + 1) *
			blockDim.x * ALPHA + threadIdx.x * ALPHA;
}

#endif /* SWKERNEL_CUH_ */
