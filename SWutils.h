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
} LaunchConfig;

typedef struct {
    int match;
    int mismatch;
    int first;
    int extension;
} scoring;

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

void getLaunchConfig(int shorterSeqLength);

void exitWithMsg(const char *msg, int exitCode);

void safeAPIcall(cudaError_t err);

scoring initScoringValues(const char *match, const char *mismath,
		const char *first, const char *extension);

#endif /* SWUTILS_H_ */
