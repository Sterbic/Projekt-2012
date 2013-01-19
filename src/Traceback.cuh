#ifndef TRACEBACK_CUH_
#define TRACEBACK_CUH_

#include "Defines.h"
#include "SWkernel.cuh"

__global__ void tracebackShort(
		int dk,
		HorizontalBuffer hbuffer,
		VerticalBuffer vbuffer,
		char *first,
		int firstLength,
		char *second,
		int secondLength,
		scoring values,
		int2 *vBusOut,
		bool gap
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
		initReverseK(&iBuffer, i, j, &hbuffer, &vbuffer, values, gap);

	for(int innerDiagonal = 0; innerDiagonal < blockDim.x; ++innerDiagonal) {

		__syncthreads();

		if(i >= 0 && i < firstLength) {
			char columnBase = second[j];

			int matchMismatch = values.mismatch;
			if(columnBase == STAGE_2_PADDING || rowBuffer.w == STAGE_2_PADDING)
				matchMismatch = 0;
			else if(rowBuffer.w == columnBase)
				matchMismatch = values.match;


			iBuffer.curr0.y = max(iBuffer.left0.y + values.extension, iBuffer.left0.x + values.first);
			iBuffer.curr0.z = max(iBuffer.up.y + values.extension, iBuffer.up.x + values.first);
			iBuffer.curr0.x = max(iBuffer.curr0.y, max(iBuffer.curr0.z, iBuffer.diagonal + matchMismatch));

			matchMismatch = values.mismatch;
			if(columnBase == STAGE_2_PADDING || rowBuffer.x == STAGE_2_PADDING)
				matchMismatch = 0;
			else if(rowBuffer.x == columnBase)
				matchMismatch = values.match;

			iBuffer.curr1.y = max(iBuffer.left1.y + values.extension, iBuffer.left1.x + values.first);
			iBuffer.curr1.z = max(iBuffer.curr0.z + values.extension, iBuffer.curr0.x + values.first);
			iBuffer.curr1.x = max(iBuffer.curr1.y, max(iBuffer.curr1.z, iBuffer.left0.x + matchMismatch));

			matchMismatch = values.mismatch;
			if(columnBase == STAGE_2_PADDING || rowBuffer.y == STAGE_2_PADDING)
				matchMismatch = 0;
			else if(rowBuffer.y == columnBase)
				matchMismatch = values.match;

			iBuffer.curr2.y = max(iBuffer.left2.y + values.extension, iBuffer.left2.x + values.first);
			iBuffer.curr2.z = max(iBuffer.curr1.z + values.extension, iBuffer.curr1.x + values.first);
			iBuffer.curr2.x = max(iBuffer.curr2.y, max(iBuffer.curr2.z, iBuffer.left1.x + matchMismatch));

			matchMismatch = values.mismatch;
			if(columnBase == STAGE_2_PADDING || rowBuffer.z == STAGE_2_PADDING)
				matchMismatch = 0;
			else if(rowBuffer.z == columnBase)
				matchMismatch = values.match;

			iBuffer.curr3.y = max(iBuffer.left3.y + values.extension, iBuffer.left3.x + values.first);
			iBuffer.curr3.z = max(iBuffer.curr2.z + values.extension, iBuffer.curr2.x + values.first);
			iBuffer.curr3.x = max(iBuffer.curr3.y, max(iBuffer.curr3.z, iBuffer.left2.x + matchMismatch));

			if(threadIdx.x < blockDim.x - 1) {
				iHbuffer[threadIdx.x + 1].x = iBuffer.curr3.x;
				iHbuffer[threadIdx.x + 1].y = iBuffer.curr3.z;
			}
			else {
				hbuffer.up[j].x = iBuffer.curr3.x;
				hbuffer.up[j].y = iBuffer.curr3.z;
			}
		}

		++j;

		__syncthreads();

		if(j == secondLength) {
			if(i >= 0 && i < firstLength) {
				dumpToVBusOut(vBusOut, &iBuffer, i);
			//	printf("ShortDumping: j = %d, i = %d, %d %d %d %d\n", j, i,
			//			iBuffer.left0.x, iBuffer.left1.x, iBuffer.left2.x, iBuffer.left3.x);

			}

			j = 0;
			i += gridDim.x * ALPHA * blockDim.x;
			getRowBuffer(i, first, &rowBuffer);
			initReverseK(&iBuffer, i, j, iHbuffer, &vbuffer, values, gap);
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
}

__global__ void tracebackLong(
		int dk,
		HorizontalBuffer hbuffer,
		VerticalBuffer vbuffer,
		char *first,
		int firstLength,
		char *second,
		int secondLength,
		scoring values,
		int2 *vBusOut,
		bool gap
		) {

	extern __shared__ int2 iHbuffer[];

	int C = secondLength / gridDim.x;

	int i = getRow(dk);
	int j = getColumn(secondLength) + blockDim.x;

	char4 rowBuffer;
	getRowBuffer(i, first, &rowBuffer);

	K iBuffer;
	if(i >= 0 && i < firstLength)
		initReverseK(&iBuffer, i, j, &hbuffer, &vbuffer, values, gap);

	for(int innerDiagonal = blockDim.x; innerDiagonal < C; ++innerDiagonal) {

		__syncthreads();

		if(i >= 0 && i < firstLength) {
			char columnBase = second[j];

			int matchMismatch = values.mismatch;
			if(columnBase == STAGE_2_PADDING || rowBuffer.w == STAGE_2_PADDING)
				matchMismatch = 0;
			else if(rowBuffer.w == columnBase)
				matchMismatch = values.match;

			iBuffer.curr0.y = max(iBuffer.left0.y + values.extension, iBuffer.left0.x + values.first);
			iBuffer.curr0.z = max(iBuffer.up.y + values.extension, iBuffer.up.x + values.first);
			iBuffer.curr0.x = max(iBuffer.curr0.y, max(iBuffer.curr0.z, iBuffer.diagonal + matchMismatch));

			matchMismatch = values.mismatch;
			if(columnBase == STAGE_2_PADDING || rowBuffer.x == STAGE_2_PADDING)
				matchMismatch = 0;
			else if(rowBuffer.x == columnBase)
				matchMismatch = values.match;

			iBuffer.curr1.y = max(iBuffer.left1.y + values.extension, iBuffer.left1.x + values.first);
			iBuffer.curr1.z = max(iBuffer.curr0.z + values.extension, iBuffer.curr0.x + values.first);
			iBuffer.curr1.x = max(iBuffer.curr1.y, max(iBuffer.curr1.z, iBuffer.left0.x + matchMismatch));

			matchMismatch = values.mismatch;
			if(columnBase == STAGE_2_PADDING || rowBuffer.y == STAGE_2_PADDING)
				matchMismatch = 0;
			else if(rowBuffer.y == columnBase)
				matchMismatch = values.match;

			iBuffer.curr2.y = max(iBuffer.left2.y + values.extension, iBuffer.left2.x + values.first);
			iBuffer.curr2.z = max(iBuffer.curr1.z + values.extension, iBuffer.curr1.x + values.first);
			iBuffer.curr2.x = max(iBuffer.curr2.y, max(iBuffer.curr2.z, iBuffer.left1.x + matchMismatch));

			matchMismatch = values.mismatch;
			if(columnBase == STAGE_2_PADDING || rowBuffer.z == STAGE_2_PADDING)
				matchMismatch = 0;
			else if(rowBuffer.z == columnBase)
				matchMismatch = values.match;

			iBuffer.curr3.y = max(iBuffer.left3.y + values.extension, iBuffer.left3.x + values.first);
			iBuffer.curr3.z = max(iBuffer.curr2.z + values.extension, iBuffer.curr2.x + values.first);
			iBuffer.curr3.x = max(iBuffer.curr3.y, max(iBuffer.curr3.z, iBuffer.left2.x + matchMismatch));

			if(threadIdx.x < blockDim.x - 1) {
				iHbuffer[threadIdx.x + 1].x = iBuffer.curr3.x;
				iHbuffer[threadIdx.x + 1].y = iBuffer.curr3.z;
			}
			else {
				hbuffer.up[j].x = iBuffer.curr3.x;
				hbuffer.up[j].y = iBuffer.curr3.z;
			}
		}

		++j;

		__syncthreads();

		int2 newUp;
		if(threadIdx.x > 0)
			newUp = iHbuffer[threadIdx.x];
		else
			newUp = hbuffer.up[j];

		pushForwardK(&iBuffer, newUp);
	}

	if (i >= 0 && i < firstLength) {

		//printf("PreDumping: j = %d, i = %d, FL = %d, SL = %d\n", j, i, firstLength, secondLength);
		if(j == secondLength) {
			//printf("LongDumping: j = %d, i = %d, %d %d %d %d\n", j, i,
				//	iBuffer.left0.x, iBuffer.left1.x, iBuffer.left2.x, iBuffer.left3.x);
			dumpToVBusOut(vBusOut, &iBuffer, i);
		}

		updateVerticalBuffer(&iBuffer, &vbuffer, i);
		if(threadIdx.x < blockDim.x -1)
			hbuffer.up[j - 1] = iHbuffer[threadIdx.x + 1];
	}
}

/* ####################################################################### */

__global__ void tracebackLastShort(
		int dk,
		HorizontalBuffer hbuffer,
		VerticalBuffer vbuffer,
		char *first,
		int firstLength,
		char *second,
		int secondLength,
		scoring values,
		bool gap,
		TracebackScore *last,
		int targetScore
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
		initReverseK(&iBuffer, i, j, &hbuffer, &vbuffer, values, gap);

	for(int innerDiagonal = 0; innerDiagonal < blockDim.x; ++innerDiagonal) {

		__syncthreads();

		if(i >= 0 && i < firstLength) {
			char columnBase = second[j];

			/*if(i == 572 && j < 600) {
				printf("i = %d, j = %d\n", i, j);
				printf("rowBase = %c %c %c %c\n", rowBuffer.w, rowBuffer.x, rowBuffer.y, rowBuffer.z);
				printf("colBase = %c\n", columnBase);
			}*/

			int matchMismatch = values.mismatch;
			if(rowBuffer.w == columnBase)
				matchMismatch = values.match;

			iBuffer.curr0.y = max(iBuffer.left0.y + values.extension, iBuffer.left0.x + values.first);
			iBuffer.curr0.z = max(iBuffer.up.y + values.extension, iBuffer.up.x + values.first);
			iBuffer.curr0.x = max(iBuffer.curr0.y, max(iBuffer.curr0.z, iBuffer.diagonal + matchMismatch));

		/*	if(i == 0) {
				printf("%d %d %d\n", iBuffer.diagonal, iBuffer.left0.y, iBuffer.up.y);
				printf("y = %d, z = %d, x = %d\n", iBuffer.curr0.y, iBuffer.curr0.z, iBuffer.curr0.x);
			}*/

			matchMismatch = values.mismatch;
			if(rowBuffer.x == columnBase)
				matchMismatch = values.match;

			iBuffer.curr1.y = max(iBuffer.left1.y + values.extension, iBuffer.left1.x + values.first);
			iBuffer.curr1.z = max(iBuffer.curr0.z + values.extension, iBuffer.curr0.x + values.first);
			iBuffer.curr1.x = max(iBuffer.curr1.y, max(iBuffer.curr1.z, iBuffer.left0.x + matchMismatch));

			matchMismatch = values.mismatch;
			if(rowBuffer.y == columnBase)
				matchMismatch = values.match;

			iBuffer.curr2.y = max(iBuffer.left2.y + values.extension, iBuffer.left2.x + values.first);
			iBuffer.curr2.z = max(iBuffer.curr1.z + values.extension, iBuffer.curr1.x + values.first);
			iBuffer.curr2.x = max(iBuffer.curr2.y, max(iBuffer.curr2.z, iBuffer.left1.x + matchMismatch));

			matchMismatch = values.mismatch;
			if(rowBuffer.z == columnBase)
				matchMismatch = values.match;

			iBuffer.curr3.y = max(iBuffer.left3.y + values.extension, iBuffer.left3.x + values.first);
			iBuffer.curr3.z = max(iBuffer.curr2.z + values.extension, iBuffer.curr2.x + values.first);
			iBuffer.curr3.x = max(iBuffer.curr3.y, max(iBuffer.curr3.z, iBuffer.left2.x + matchMismatch));

			checkStartPoint(&iBuffer, i, j, targetScore, last);

			if(threadIdx.x < blockDim.x - 1) {
				iHbuffer[threadIdx.x + 1].x = iBuffer.curr3.x;
				iHbuffer[threadIdx.x + 1].y = iBuffer.curr3.z;
			}
			else {
				hbuffer.up[j].x = iBuffer.curr3.x;
				hbuffer.up[j].y = iBuffer.curr3.z;
			}
		}

		++j;

		__syncthreads();

		if(j == secondLength) {
			j = 0;
			i += gridDim.x * ALPHA * blockDim.x;
			getRowBuffer(i, first, &rowBuffer);
			initReverseK(&iBuffer, i, j, iHbuffer, &vbuffer, values, gap);
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
}

__global__ void tracebackLastLong(
		int dk,
		HorizontalBuffer hbuffer,
		VerticalBuffer vbuffer,
		char *first,
		int firstLength,
		char *second,
		int secondLength,
		scoring values,
		bool gap,
		TracebackScore *last,
		int targetScore
		) {

	extern __shared__ int2 iHbuffer[];

	int C = secondLength / gridDim.x;

	int i = getRow(dk);
	int j = getColumn(secondLength) + blockDim.x;

	char4 rowBuffer;
	getRowBuffer(i, first, &rowBuffer);

	K iBuffer;
	if(i >= 0 && i < firstLength)
		initReverseK(&iBuffer, i, j, &hbuffer, &vbuffer, values, gap);

	for(int innerDiagonal = blockDim.x; innerDiagonal < C; ++innerDiagonal) {

		__syncthreads();

		if(i >= 0 && i < firstLength) {
			char columnBase = second[j];

			int matchMismatch = values.mismatch;
			if(rowBuffer.w == columnBase)
				matchMismatch = values.match;

			iBuffer.curr0.y = max(iBuffer.left0.y + values.extension, iBuffer.left0.x + values.first);
			iBuffer.curr0.z = max(iBuffer.up.y + values.extension, iBuffer.up.x + values.first);
			iBuffer.curr0.x = max(iBuffer.curr0.y, max(iBuffer.curr0.z, iBuffer.diagonal + matchMismatch));

			matchMismatch = values.mismatch;
			if(rowBuffer.x == columnBase)
				matchMismatch = values.match;

			iBuffer.curr1.y = max(iBuffer.left1.y + values.extension, iBuffer.left1.x + values.first);
			iBuffer.curr1.z = max(iBuffer.curr0.z + values.extension, iBuffer.curr0.x + values.first);
			iBuffer.curr1.x = max(iBuffer.curr1.y, max(iBuffer.curr1.z, iBuffer.left0.x + matchMismatch));

			matchMismatch = values.mismatch;
			if(rowBuffer.y == columnBase)
				matchMismatch = values.match;

			iBuffer.curr2.y = max(iBuffer.left2.y + values.extension, iBuffer.left2.x + values.first);
			iBuffer.curr2.z = max(iBuffer.curr1.z + values.extension, iBuffer.curr1.x + values.first);
			iBuffer.curr2.x = max(iBuffer.curr2.y, max(iBuffer.curr2.z, iBuffer.left1.x + matchMismatch));

			matchMismatch = values.mismatch;
			if(rowBuffer.z == columnBase)
				matchMismatch = values.match;

			iBuffer.curr3.y = max(iBuffer.left3.y + values.extension, iBuffer.left3.x + values.first);
			iBuffer.curr3.z = max(iBuffer.curr2.z + values.extension, iBuffer.curr2.x + values.first);
			iBuffer.curr3.x = max(iBuffer.curr3.y, max(iBuffer.curr3.z, iBuffer.left2.x + matchMismatch));

			checkStartPoint(&iBuffer, i, j, targetScore, last);

			if(threadIdx.x < blockDim.x - 1) {
				iHbuffer[threadIdx.x + 1].x = iBuffer.curr3.x;
				iHbuffer[threadIdx.x + 1].y = iBuffer.curr3.z;
			}
			else {
				hbuffer.up[j].x = iBuffer.curr3.x;
				hbuffer.up[j].y = iBuffer.curr3.z;
			}
		}

		++j;

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
}

#endif /* TRACEBACK_CUH_ */
