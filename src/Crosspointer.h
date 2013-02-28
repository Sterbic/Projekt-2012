#ifndef CROSSPOINTER_H_
#define CROSSPOINTER_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <string.h>

#include "Defines.h"
#include "SWutils.h"
#include "FASTA.h"

class Crosspointer {
private:
	int srHeight;
	SWquery *query;
	CUDAcard *gpu;
	scoring *values;
	int srIndex;
	char fileName[50];
	LaunchConfig stdLaunchConfig;
	TracebackScore target;
	std::vector<TracebackScore> xPoints;

	VerticalBuffer vBuffer;
	HorizontalBuffer hBuffer;
	int2 *hBufferUp;
	char *devRow;
	char *devColumn;
	int2 *vBusOut;
	int2 *devVBusOut;
	char *firstReversed;
	char *secondReversed;
	bool gap;
	int2 *specialRow;
	char *pad;
	int D;
	int readOffset;
	int heightOffset;
	int widthOffset;

	void doStage2Cross();
	bool findXtoFirstSR();
	void findStartX();
	void findXonSpecRows();

	void reInitHBuffer();
	void prepareFileName();
	bool foundLast(TracebackScore *last);
	void doPadding(char *devPtr, int get);

public:
	Crosspointer(SWquery *query, CUDAcard *gpu, scoring *values, alignmentScore endPoint, int srHeight);

	~Crosspointer();

	std::vector<TracebackScore> findCrosspoints();

	LaunchConfig getStdLaunchConfig();
	HorizontalBuffer getHBuffer();
	VerticalBuffer getVBuffer();
	char *getDevRow();
	char *getDevColumn();
	int getSrHeight();
	bool getGap();
	scoring getValues();
	TracebackScore getTarget();
	int2 *getVBusOut();
};

extern "C" void kernelWrapperTB(Crosspointer *xPointer, int dk, TracebackScore *devLast, int kernel);

#endif
