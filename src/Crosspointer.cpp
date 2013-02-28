#include "Crosspointer.h"

Crosspointer::Crosspointer(SWquery *query, CUDAcard *gpu, scoring *values, alignmentScore endPoint, int srHeight) {
	this->query = query;

	this->srHeight = srHeight;

	this->gpu = gpu;

	this->values = values;

	this->target.column = endPoint.column;
	this->target.row = endPoint.row;
	this->target.score = endPoint.score;
	this->target.gap = false;

	TracebackScore last = this->target;
	this->xPoints.push_back(last);

	hBufferUp = (int2 *) malloc(sizeof(int2) * srHeight);
	if(hBufferUp == NULL)
	    exitWithMsg("Allocation error hBufferUp", -1);

	for(int i = 0; i < srHeight; i++)
		hBufferUp[i].x = hBufferUp[i].y = -1000000000;

	hBuffer.up = (int2 *) cudaGetDeviceCopy(hBufferUp, sizeof(int2) * srHeight);

	srIndex = (target.row / srHeight) * srHeight - 1;

	stdLaunchConfig = getLaunchConfig(srHeight, *gpu);

	initVerticalBuffer(&vBuffer, stdLaunchConfig);

	devRow = (char *) cudaGetSpaceAndSet(srHeight * sizeof(char), 0);
	devColumn = (char *) cudaGetSpaceAndSet(srHeight * sizeof(char), 0);

	vBusOut = (int2 *) malloc(srHeight * sizeof(int2));
	if(vBusOut == NULL)
		exitWithMsg("Allocation error vBusOut", -1);

	this->devVBusOut = (int2 *) cudaGetSpaceAndSet(srHeight * sizeof(int2), 0);

	firstReversed = query->getFirst()->getReversedSequence(target.row);
	secondReversed = query->getSecond()->getReversedSequence(target.column);

	gap = false;

	specialRow = (int2 *) malloc(query->getSecond()->getPaddedLength() * sizeof(int2));
	if(specialRow == NULL)
		exitWithMsg("Error allocating special row.", -1);

	D = stdLaunchConfig.blocks + ceil(((double) srHeight) / (ALPHA * stdLaunchConfig.threads)) - 1;

	readOffset = 0;
	heightOffset = 0;
	widthOffset = 0;

	pad = (char *) malloc(srHeight * sizeof(char));
	if(pad == NULL)
		exitWithMsg("Error allocating pad array.", -1);

	memset(pad, STAGE_2_PADDING, srHeight * sizeof(char));
}

Crosspointer::~Crosspointer() {
	if(hBufferUp != NULL) free(hBufferUp);
	if(hBuffer.up != NULL) safeAPIcall(cudaFree(hBuffer.up), __LINE__);
	if(devRow != NULL) safeAPIcall(cudaFree(devRow), __LINE__);
	if(devColumn != NULL) safeAPIcall(cudaFree(devColumn), __LINE__);
	if(vBusOut != NULL) free(vBusOut);
	if(devVBusOut != NULL) safeAPIcall(cudaFree(devVBusOut), __LINE__);
	if(specialRow != NULL) free(specialRow);
	if(pad != NULL) free(pad);
}

void Crosspointer::reInitHBuffer() {
	safeAPIcall(cudaMemcpy(hBuffer.up, hBufferUp, sizeof(int2) * srHeight, cudaMemcpyHostToDevice), __LINE__);
}

void Crosspointer::prepareFileName() {
	memset(fileName, 0, 50);
	sprintf(fileName, "temp/row_%d", srIndex + 1);
}

std::vector<TracebackScore> Crosspointer::findCrosspoints() {
	doStage2Cross();
	return xPoints;
}

void Crosspointer::doStage2Cross() {
	if(findXtoFirstSR()) findXonSpecRows();
	//findStartX();
}

void Crosspointer::findXonSpecRows() {

}

void Crosspointer::doPadding(char *devPtr, int get) {
	safeAPIcall(cudaMemcpy(devPtr + get, pad, (srHeight - get) * sizeof(char),
			cudaMemcpyHostToDevice), __LINE__);
}

bool Crosspointer::findXtoFirstSR() {
	prepareFileName();

	FILE *f = fopen(fileName, "rb");
	if(f == NULL)
		exitWithMsg("Error opening special row file.", -1);

	fread(specialRow, sizeof(int2), query->getSecond()->getPaddedLength(), f);
	fclose(f);

	int getVertical = target.row - srIndex;
	int verticalPadding = srHeight - getVertical;
	printf("getV = %d, vp = %d\n", getVertical, verticalPadding);

	safeAPIcall(cudaMemcpy(devColumn, firstReversed + heightOffset,
			getVertical * sizeof(char), cudaMemcpyHostToDevice), __LINE__);

	if(getVertical < srIndex) doPadding(devColumn, getVertical);
	//printf("widthOff = %d, ts = %d\n", widthOffset, target.column);

	while(widthOffset < target.column) {
		int getHorizontal = std::min(srHeight - verticalPadding, target.column - widthOffset);
		printf("srh = %d, vp = %d, tc = %d, wo = %d\n", srHeight, verticalPadding, target.column, widthOffset);

		safeAPIcall(cudaMemcpy(devRow, secondReversed + widthOffset,
				getHorizontal * sizeof(char), cudaMemcpyHostToDevice), __LINE__);

		if(getHorizontal < srHeight) doPadding(devRow, getHorizontal);

		for(int dk = 0; dk < D + stdLaunchConfig.blocks; ++dk) {
			kernelWrapperTB(this, dk, NULL, TRACEBACK_SHORT_LONG);
		}

		memset(vBusOut, 0, srHeight * sizeof(int2));
		printf("vPad = %d, getH = %d\n", verticalPadding, getHorizontal);
		printf("devvbou %d, vbo %d, offset = %d\n", vBusOut, devVBusOut, srHeight - getHorizontal);

		safeAPIcall(cudaMemcpy(vBusOut, devVBusOut, srHeight * sizeof(int2), cudaMemcpyDeviceToHost), __LINE__);

		TracebackScore tracebackScore = getTracebackScore(
				*values, srIndex - 1, getHorizontal, vBusOut,
				specialRow + target.column - widthOffset - getHorizontal - 1,
				target.score, target.column - widthOffset);

		if(tracebackScore.column != -1) {
			target = tracebackScore;
			gap = tracebackScore.gap;

			printf("Crosspoint [%d, %d] = %d\n", target.row, target.column, target.score);
			xPoints.push_back(tracebackScore);

			srIndex -= srHeight;
			heightOffset += getVertical;

			reInitHBuffer();

			return true;
		}
		else {
			widthOffset += getHorizontal;
		}
	}

	widthOffset = heightOffset = 0;
	return false;
}

void Crosspointer::findStartX() {
	if(target.score == 0) return;
	printf("Starting last with target score %d\n", target.score);

	int lastSize = stdLaunchConfig.blocks * stdLaunchConfig.threads * sizeof(TracebackScore);
	TracebackScore *last = (TracebackScore *) malloc(lastSize);
	if(last == NULL)
		exitWithMsg("Allocation error for traceback last", -1);

	TracebackScore *devLast = (TracebackScore *) cudaGetSpaceAndSet(lastSize, -1);

	char padLastRows[240];
	memset(padLastRows, STAGE_2_PADDING_LAST_ROWS, 240);

	int getVertical = std::min(srHeight, target.row + 1);
	printf("getV = %d, srh = %d\n", getVertical, srHeight);
	safeAPIcall(cudaMemcpy(devColumn, firstReversed + heightOffset,
	    		getVertical * sizeof(char), cudaMemcpyHostToDevice), __LINE__);

	for(int i = getVertical; i < srHeight - getVertical; i += 240) {
				safeAPIcall(cudaMemcpy(devColumn + i, padLastRows,
						std::min(srHeight - i, 240) * sizeof(char), cudaMemcpyHostToDevice), __LINE__);
	}

	bool found = false;

	while(widthOffset < target.column) {
		int getHorizontal = std::min(srHeight, target.column - widthOffset + 1);
		safeAPIcall(cudaMemcpy(devRow, secondReversed + widthOffset,
				getHorizontal * sizeof(char), cudaMemcpyHostToDevice), __LINE__);

		for(int i = getHorizontal; i < srHeight - getHorizontal; i += 240) {
			safeAPIcall(cudaMemcpy(devRow + i, pad,
					std::min(srHeight - i, 240) * sizeof(char), cudaMemcpyHostToDevice), __LINE__);
		}

		for(int dk = 0; dk < D + stdLaunchConfig.blocks; dk++) {
			//kernel call
			kernelWrapperTB(this, dk, devLast, TRACEBACK_LAST_SHORT);

			safeAPIcall(cudaMemcpy(last, devLast, lastSize, cudaMemcpyDeviceToHost), __LINE__);
			found = foundLast(last);
			if(found) break;

			// kernel call
			kernelWrapperTB(this, dk, devLast, TRACEBACK_LAST_LONG);

			safeAPIcall(cudaMemcpy(last, devLast, lastSize, cudaMemcpyDeviceToHost), __LINE__);
			found = foundLast(last);
			if(found) break;

		}

		if(found) break;
		widthOffset += getHorizontal;
	}

	free(last);
	safeAPIcall(cudaFree(devLast), __LINE__);
}

bool Crosspointer::foundLast(TracebackScore *last) {
	for(int i = 0; i < stdLaunchConfig.blocks * stdLaunchConfig.threads; i++) {
		if(last[i].score != -1) {
			TracebackScore lastScore;
			lastScore.score = last[i].score;
			lastScore.row = target.row - last[i].row;
			lastScore.column = target.column - last[i].column - widthOffset;
			lastScore.gap = false;

			xPoints.push_back(lastScore);

			printf("Found last: [%d, %d] = %d\n", lastScore.row, lastScore.column, lastScore.score);

			return true;
		}
	}

	return false;
}

LaunchConfig Crosspointer::getStdLaunchConfig() {
	return stdLaunchConfig;
}

HorizontalBuffer Crosspointer::getHBuffer() {
	return hBuffer;
}

VerticalBuffer Crosspointer::getVBuffer() {
	return vBuffer;
}

char *Crosspointer::getDevRow() {
	return devRow;
}

char *Crosspointer::getDevColumn() {
	return devColumn;
}

int Crosspointer::getSrHeight() {
	return srHeight;
}

bool Crosspointer::getGap() {
	return gap;
}

scoring Crosspointer::getValues() {
	return *values;
}

TracebackScore Crosspointer::getTarget() {
	return target;
}

int2 *Crosspointer::getVBusOut() {
	return vBusOut;
}
