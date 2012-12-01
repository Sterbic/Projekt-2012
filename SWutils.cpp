/*
 * SWutils.cpp
 *
 *  Created on: Dec 1, 2012
 *      Author: Luka
 */

#include <stdio.h>
#include <stdlib.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "SWutils.h"

void exitWithMsg(const char *msg, int exitCode) {
	printf("ERROR\n");
	printf("%s\n\n", msg);
	exit(exitCode);
}

void safeAPIcall(cudaError_t err) {
	if(err != cudaSuccess)
		exitWithMsg(cudaGetErrorString(err), -2);
}

scoring initScoringValues(const char *match, const char *mismath,
		const char *first, const char *extension) {

	scoring values;
	printf("Initializing scoring values... ");
	values.match = atoi(match);
	values.mismatch = atoi(mismath);
	values.first = atoi(first);
	values.extension = atoi(extension);

	if(values.match < 1 || values.mismatch > -1 || values.first > -1 || values.extension > -1)
		exitWithMsg("One or more scoring values were not usable!", -1);
	else {
		printf("DONE\n\nScoring values:\n");
		printf("	>Match: %d\n", values.match);
		printf("	>Mismatch: %d\n", values.mismatch);
		printf("	>First gap: %d\n", values.first);
		printf("	>Gap extension: %d\n\n", values.extension);
	}

	return values;
}

cudaTimer::cudaTimer() {
	cudaEventCreate(&_start);
	cudaEventCreate(&_stop);
}

cudaTimer::~cudaTimer() {
	cudaEventDestroy(_start);
	cudaEventDestroy(_stop);
}

void cudaTimer::start() {
	cudaEventRecord(_start, 0);
}

void cudaTimer::stop() {
	cudaEventRecord(_stop, 0);
	cudaEventSynchronize(_stop);
}

float cudaTimer::getElapsedTimeMillis() {
	float time;
	cudaEventElapsedTime(&time, _start, _stop);
	return time;
}
