#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>

#include "Defines.h"
#include "Builders.h"

RowBuilder::RowBuilder(int firstLen, int secondLen, LaunchConfig *config) {
	this->firstLen = firstLen;
	this->secondLen = secondLen;
	this->config = config;

	this->rowHeight = ALPHA * config->threads;

	hb = (int2 *) malloc(secondLen * sizeof(int2));
	if(hb == NULL)
		exitWithMsg("Allocation error in row builder.", -1);
}

RowBuilder::~RowBuilder() {
	if(hb != NULL) free(hb);
}

void RowBuilder::dumpShort(int2 *devHBuffer, int dk) {
	safeAPIcall(cudaMemcpy(hb, devHBuffer, secondLen * sizeof(int2), cudaMemcpyDeviceToHost), __LINE__);

	int C = secondLen / config->blocks;
	int row = (dk + 1) * ALPHA * config->threads;
	char fileName[50];

	int offset = 0;
	int count = 1;

	for (int counter = dk; counter >= 0; --counter) {
		if(row % rowHeight == 0 && row >= 0 && row < firstLen) {
			memset(fileName, 0, 50);
			sprintf(fileName, "temp/row_%d", row);
			FILE *f = fopen(fileName, "ab");
			sprintf(fileName, "temp/row_%d.txt", row);
			FILE *tmp = fopen(fileName, "a");

			printf("C = %d, Count = %d, offset = %d, counter = %d\n",
						C, count, offset, counter);

			if(offset == secondLen - config->threads + 1)
				count--;

			fwrite(hb + offset, sizeof(int2), config->threads, f);

			for (int i = 0; i < count; ++i) {
				fprintf(tmp, "%d %d\n", (hb + offset + i)->x, (hb + offset + i)->y);
			}
			printf("writing done\n");

			fclose(f);
			fclose(tmp);
		}

		offset += count + C - config->threads;
		row -= ALPHA * config->threads;
	}
}

void RowBuilder::dumpLong(int2 *devHBuffer, int dk) {
	safeAPIcall(cudaMemcpy(hb, devHBuffer, secondLen * sizeof(int2), cudaMemcpyDeviceToHost), __LINE__);

	int C = secondLen / config->blocks;
	int row = (dk + 1) * ALPHA * config->threads;
	char fileName[50];

	int offset = 1;
	for (int counter = dk; counter >= 0; --counter) {

		if(row % rowHeight == 0 && row >= 0 && row < firstLen) {
			memset(fileName, 0, 50);
			sprintf(fileName, "temp/row_%d", row);
			FILE *f = fopen(fileName, "ab");

			sprintf(fileName, "temp/row_%d.txt", row);
			FILE *tmp = fopen(fileName, "a");

			printf("Count = %d, offset = %d, counter = %d\n",
					C - config->threads, offset, counter);

			fwrite(hb + offset, sizeof(int2), C - config->threads, f);

			for (int i = 0; i < C - config->threads; ++i) {
				fprintf(tmp, "%d %d\n", (hb + offset + i)->x, (hb + offset + i)->y);
			}

			printf("writing done long\n");

			fclose(f);
			fclose(tmp);
		}

		offset += C;
		row -= ALPHA * config->threads;
	}
}

int RowBuilder::getRowHeight(void) {
	return rowHeight;
}
