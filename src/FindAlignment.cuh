#ifndef FINDALIGNMENT_CUH_
#define FINDALIGNMENT_CUH_

#include "Defines.h"
#include "SWkernel.cuh"

texture<char> texSecond;

__global__ void shortPhase(
		int dk,
		HorizontalBuffer hbuffer,
		VerticalBuffer vbuffer,
		char *first,
		int firstLength,
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

	int3 localMax;
	localMax.x = 0;

	for(int innerDiagonal = 0; innerDiagonal < blockDim.x; innerDiagonal++) {

		__syncthreads();

		if(i >= 0 && i < firstLength) {
			char columnBase = tex1Dfetch(texSecond, j);

			int matchMismatch = values.mismatch;
			if(rowBuffer.w == columnBase)
				matchMismatch = values.match;

			iBuffer.curr0.y = max(iBuffer.left0.y + values.extension, iBuffer.left0.x + values.first);
			iBuffer.curr0.z = max(iBuffer.up.y + values.extension, iBuffer.up.x + values.first);
			iBuffer.curr0.x = max(max(0, iBuffer.curr0.y), max(iBuffer.curr0.z, iBuffer.diagonal + matchMismatch));

			if(iBuffer.curr0.x > localMax.x) {
				localMax.x = iBuffer.curr0.x;
				localMax.y = i;
				localMax.z = j;
			}

			matchMismatch = values.mismatch;
			if(rowBuffer.x == columnBase)
				matchMismatch = values.match;

			iBuffer.curr1.y = max(iBuffer.left1.y + values.extension, iBuffer.left1.x + values.first);
			iBuffer.curr1.z = max(iBuffer.curr0.z + values.extension, iBuffer.curr0.x + values.first);
			iBuffer.curr1.x = max(max(0, iBuffer.curr1.y), max(iBuffer.curr1.z, iBuffer.left0.x + matchMismatch));

			if(iBuffer.curr1.x > localMax.x) {
				localMax.x = iBuffer.curr1.x;
				localMax.y = i + 1;
				localMax.z = j;
			}

			matchMismatch = values.mismatch;
			if(rowBuffer.y == columnBase)
				matchMismatch = values.match;

			iBuffer.curr2.y = max(iBuffer.left2.y + values.extension, iBuffer.left2.x + values.first);
			iBuffer.curr2.z = max(iBuffer.curr1.z + values.extension, iBuffer.curr1.x + values.first);
			iBuffer.curr2.x = max(max(0, iBuffer.curr2.y), max(iBuffer.curr2.z, iBuffer.left1.x + matchMismatch));

			if(iBuffer.curr2.x > localMax.x) {
				localMax.x = iBuffer.curr2.x;
				localMax.y = i + 2;
				localMax.z = j;
			}

			matchMismatch = values.mismatch;
			if(rowBuffer.z == columnBase)
				matchMismatch = values.match;

			iBuffer.curr3.y = max(iBuffer.left3.y + values.extension, iBuffer.left3.x + values.first);
			iBuffer.curr3.z = max(iBuffer.curr2.z + values.extension, iBuffer.curr2.x + values.first);
			iBuffer.curr3.x = max(max(0, iBuffer.curr3.y), max(iBuffer.curr3.z, iBuffer.left2.x + matchMismatch));

			if(iBuffer.curr3.x > localMax.x) {
				localMax.x = iBuffer.curr3.x;
				localMax.y = i + 3;
				localMax.z = j;
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

		if(j == secondLength) {
			j = 0;
			i += gridDim.x * ALPHA * blockDim.x;
			//rowBuffer = tex1Dfetch(texFirst, i / 4);
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
	}

	if (i >= 0 && i < firstLength) {
		updateVerticalBuffer(&iBuffer, &vbuffer, i);
		if(threadIdx.x < blockDim.x -1)
			hbuffer.up[j - 1] = iHbuffer[threadIdx.x + 1];
	}

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(localMax.x > score[index].score) {
		score[index].score = localMax.x;
		score[index].row = localMax.y;
		score[index].column = localMax.z;
	}
}

__global__ void longPhase(
		int dk,
		HorizontalBuffer hbuffer,
		VerticalBuffer vbuffer,
		char *first,
		int firstLength,
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

	int3 localMax;
	localMax.x = 0;

	for(int innerDiagonal = blockDim.x; innerDiagonal < C; innerDiagonal++) {

		__syncthreads();

		if(i >= 0 && i < firstLength) {
			char columnBase = tex1Dfetch(texSecond, j);

			int matchMismatch = values.mismatch;
			if(rowBuffer.w == columnBase)
				matchMismatch = values.match;

			iBuffer.curr0.y = max(iBuffer.left0.y + values.extension, iBuffer.left0.x + values.first);
			iBuffer.curr0.z = max(iBuffer.up.y + values.extension, iBuffer.up.x + values.first);
			iBuffer.curr0.x = max(max(0, iBuffer.curr0.y), max(iBuffer.curr0.z, iBuffer.diagonal + matchMismatch));

			if(iBuffer.curr0.x > localMax.x) {
				localMax.x = iBuffer.curr0.x;
				localMax.y = i;
				localMax.z = j;
			}

			matchMismatch = values.mismatch;
			if(rowBuffer.x == columnBase)
				matchMismatch = values.match;

			iBuffer.curr1.y = max(iBuffer.left1.y + values.extension, iBuffer.left1.x + values.first);
			iBuffer.curr1.z = max(iBuffer.curr0.z + values.extension, iBuffer.curr0.x + values.first);
			iBuffer.curr1.x = max(max(0, iBuffer.curr1.y), max(iBuffer.curr1.z, iBuffer.left0.x + matchMismatch));

			if(iBuffer.curr1.x > localMax.x) {
				localMax.x = iBuffer.curr1.x;
				localMax.y = i + 1;
				localMax.z = j;
			}

			matchMismatch = values.mismatch;
			if(rowBuffer.y == columnBase)
				matchMismatch = values.match;

			iBuffer.curr2.y = max(iBuffer.left2.y + values.extension, iBuffer.left2.x + values.first);
			iBuffer.curr2.z = max(iBuffer.curr1.z + values.extension, iBuffer.curr1.x + values.first);
			iBuffer.curr2.x = max(max(0, iBuffer.curr2.y), max(iBuffer.curr2.z, iBuffer.left1.x + matchMismatch));

			if(iBuffer.curr2.x > localMax.x) {
				localMax.x = iBuffer.curr2.x;
				localMax.y = i + 2;
				localMax.z = j;
			}

			matchMismatch = values.mismatch;
			if(rowBuffer.z == columnBase)
				matchMismatch = values.match;

			iBuffer.curr3.y = max(iBuffer.left3.y + values.extension, iBuffer.left3.x + values.first);
			iBuffer.curr3.z = max(iBuffer.curr2.z + values.extension, iBuffer.curr2.x + values.first);
			iBuffer.curr3.x = max(max(0, iBuffer.curr3.y), max(iBuffer.curr3.z, iBuffer.left2.x + matchMismatch));

			if(iBuffer.curr3.x > localMax.x) {
				localMax.x = iBuffer.curr3.x;
				localMax.y = i + 3;
				localMax.z = j;
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

		int2 newUp;
		if(threadIdx.x > 0)
			newUp = iHbuffer[threadIdx.x];
		else
			newUp = hbuffer.up[j];

		pushForwardK(&iBuffer, newUp);
	}

	if (i >= 0 && i < firstLength) {
		updateVerticalBuffer(&iBuffer, &vbuffer, i);
		if(threadIdx.x < blockDim.x -1)
			hbuffer.up[j - 1] = iHbuffer[threadIdx.x + 1];
	}

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(localMax.x > score[index].score) {
		score[index].score = localMax.x;
		score[index].row = localMax.y;
		score[index].column = localMax.z;
	}
}

#endif /* FINDALIGNMENT_CUH_ */
