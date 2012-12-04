/*
 * SWutils.h
 *
 *  Created on: Dec 1, 2012
 *      Author: Luka
 */

#ifndef SWUTILS_H_
#define SWUTILS_H_

#include <driver_types.h>

typedef struct {
	int blocks;
	int threads;
	int sharedMemSize;
} LaunchConfig;

typedef struct {
    int score;
    int row;
    int column;
} alignmentScore;

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

LaunchConfig getLaunchConfig(int shorterSeqLength, CUDAcard gpu);

void printLaunchConfig(LaunchConfig config);

CUDAcard findBestDevice();

void printCardInfo(CUDAcard gpu);

void exitWithMsg(const char *msg, int exitCode);

void safeAPIcall(cudaError_t err);

void *cudaGetSpaceAndSet(int size, int setTo);

void *cudaGetDeviceCopy(void *src, int size);

scoring initScoringValues(const char *match, const char *mismath,
		const char *first, const char *extension);

#endif /* SWUTILS_H_ */
