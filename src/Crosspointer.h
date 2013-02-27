#ifndef CROSSPOINTER_H_
#define CROSSPOINTER_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Defines.h"
#include "SWutils.h"
#include "FASTA.h"

class Crosspointer {
private:
	int srHeight;
	SWquery *query;
	CUDAcard *gpu;
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
	int D;
	int readOffset;

	void doStage2Cross();
	void findXtoFirstSR();
	void findStartX();
	void findXonSpecRows();

public:
	Crosspointer(SWquery *query, CUDAcard *gpu, alignmentScore endPoint, int srHeight);

	virtual ~Crosspointer();

	std::vector<TracebackScore> findCrosspoints();
};

#endif /* CROSSPOINTER_H_ */
