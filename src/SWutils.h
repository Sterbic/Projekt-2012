/*
 * SWutils.h
 *
 *  Created on: Dec 1, 2012
 *      Author: Luka
 */

#ifndef SWUTILS_H_
#define SWUTILS_H_

#include <cuda.h>
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

class SWquerry {
private:
	FASTAsequence *first;
	FASTAsequence *second;
	__align__(16) char *deviceFirst;
	__align__(16) char *deviceSecond;
	bool prepared;

	void checkPrepared() {
		if(!prepared)
			exitWithMsg("SWquerry was not ready when a get method was invoked.", -1);
	}

public:
	SWquerry(FASTAsequence *first, FASTAsequence *second);

	~SWquerry();

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

void safeAPIcall(cudaError_t err);

void *cudaGetSpaceAndSet(int size, int setTo);

void *cudaGetDeviceCopy(void *src, int size);

void initGlobalBuffer(GlobalBuffer *buffer, int secondLength, LaunchConfig config);

void freeGlobalBuffer(GlobalBuffer *buffer);

scoring initScoringValues(const char *match, const char *mismath,
		const char *first, const char *extension);

#endif /* SWUTILS_H_ */
