/*
 * SWutils.h
 *
 *  Created on: Dec 1, 2012
 *      Author: Luka
 */

#ifndef SWUTILS_H_
#define SWUTILS_H_

#include <driver_types.h>

#include "FASTA.h"

typedef struct {
	int blocks;
	int threads;
	int sharedMemSize;
} LaunchConfig;

typedef struct {
    int match;
    int mismatch;
    int first;
    int extension;
} scoring;

typedef struct {
	char name[256];
	int cardNumber;
	int cardsInSystem;
	int major;
	int minor;
	unsigned long long globalMem;
	int maxThreadsPerBlock;
	int SMs;
	int cudaCores;
} CUDAcard;

typedef struct {
	bool gap;
	int score;
	int column;
	int row;
} TracebackScore;

void exitWithMsg(const char *msg, int exitCode);

class cudaTimer {
private:
    cudaEvent_t _start;
    cudaEvent_t _stop;

public:
    cudaTimer();

    ~cudaTimer();

    void start();

    void stop();

    float getElapsedTimeMillis();
};

class SWquery {
private:
	FASTAsequence *first;
	FASTAsequence *second;
	char *deviceFirst;
	char *deviceSecond;
	bool prepared;

	void checkPrepared() {
		if(!prepared)
			exitWithMsg("SWquerry was not ready when a get method was invoked.", -1);
	}

public:
	SWquery(FASTAsequence *first, FASTAsequence *second);

	~SWquery();

	void prepare(LaunchConfig config);

	char *getDevFirst();

	char *getDevSecond();

	FASTAsequence *getFirst();

	FASTAsequence *getSecond();
};

LaunchConfig getLaunchConfig(int shorterSeqLength, CUDAcard gpu);

void printLaunchConfig(LaunchConfig config);

CUDAcard findBestDevice();

void printCardInfo(CUDAcard gpu);

void safeAPIcall(cudaError_t err, int line);

void *cudaGetSpaceAndSet(int size, int setTo);

void *cudaGetDeviceCopy(void *src, int size);

void initGlobalBuffer(GlobalBuffer *buffer, int secondLength, LaunchConfig config);

void freeGlobalBuffer(GlobalBuffer *buffer);

void initVerticalBuffer(VerticalBuffer *vBuffer, LaunchConfig config);

void freeVerticalBuffer(VerticalBuffer *vBuffer);

TracebackScore getTracebackScore(scoring values, bool frontGap, int row, int rows, int cols,
		int2 *vBusOut, int2 *specialRow, int targetScore);

alignmentScore getMaxScore(alignmentScore *score, int n);

scoring initScoringValues(const char *match, const char *mismath,
		const char *first, const char *extension);

#endif /* SWUTILS_H_ */
