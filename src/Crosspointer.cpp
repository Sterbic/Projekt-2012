#include "Crosspointer.h"

Crosspointer::Crosspointer(SWquery *query, CUDAcard *gpu, alignmentScore endPoint, int srHeight) {
	this->query = query;

	this->srHeight = srHeight;

	this->gpu = gpu;

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

	srIndex = (target.row / srHeight) * srHeight;

	stdLaunchConfig = getLaunchConfig(srHeight, *gpu);

	initVerticalBuffer(&vBuffer, stdLaunchConfig);

	devRow = (char *) cudaGetSpaceAndSet(srHeight * sizeof(char), 0);
	devColumn = (char *) cudaGetSpaceAndSet(srHeight * sizeof(char), 0);

	vBusOut = (int2 *) malloc(srHeight * sizeof(int2));
	if(vBusOut == NULL)
		exitWithMsg("Allocation error vBusOut", -1);

	devVBusOut = (int2 *) cudaGetSpaceAndSet(srHeight * sizeof(int2), 0);

	char *firstReversed = query->getFirst()->getReversedSequence(target.row);
	char *secondReversed = query->getSecond()->getReversedSequence(target.column);

	gap = false;

	specialRow = (int2 *) malloc(query->getSecond()->getPaddedLength() * sizeof(int2));
	if(specialRow == NULL)
		exitWithMsg("Error allocating special row.", -1);

	D = stdLaunchConfig.blocks + ceil(((double) srHeight) / (ALPHA * stdLaunchConfig.threads)) - 1;

	readOffset = 0;
}

Crosspointer::~Crosspointer() {
	if(hBufferUp != NULL) free(hBufferUp);
	if(hBuffer.up != NULL) safeAPIcall(cudaFree(hBuffer.up), __LINE__);
	if(devRow != NULL) safeAPIcall(cudaFree(devRow), __LINE__);
	if(devColumn != NULL) safeAPIcall(cudaFree(devColumn), __LINE__);
	if(vBusOut != NULL) free(vBusOut);
	if(devVBusOut != NULL) safeAPIcall(cudaFree(devVBusOut), __LINE__);
	if(specialRow != NULL) free(specialRow);
}

std::vector<TracebackScore> Crosspointer::findCrosspoints() {
	doStage2Cross();
	return xPoints;
}

void Crosspointer::doStage2Cross() {
	findXtoFirstSR();
	findXonSpecRows();
	findStartX();
}

void Crosspointer::findXonSpecRows() {

}

void Crosspointer::findXtoFirstSR() {

}

void Crosspointer::findStartX() {

}
