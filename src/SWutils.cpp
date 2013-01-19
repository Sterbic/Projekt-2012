#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Defines.h"
#include "SWutils.h"

void exitWithMsg(const char *msg, int exitCode) {
	printf("ERROR\n");
	printf("%s\n\n", msg);
	exit(exitCode);
}

void safeAPIcall(cudaError_t err, int line) {
	if(err != cudaSuccess) {
		printf("Error in line %d\n", line);
		exitWithMsg(cudaGetErrorString(err), -2);
	}
}

scoring initScoringValues(const char *match, const char *mismath,
		const char *first, const char *extension) {

	scoring values;
	printf("> Initializing scoring values... ");
	values.match = atoi(match);
	values.mismatch = atoi(mismath);
	values.first = atoi(first);
	values.extension = atoi(extension);

	if(values.match < 1 || values.mismatch > -1 || values.first > -1 || values.extension > -1)
		exitWithMsg("One or more scoring values were not usable!", -1);
	else {
		printf("DONE\n\nScoring values:\n");
		printf("\t>Match: %d\n", values.match);
		printf("\t>Mismatch: %d\n", values.mismatch);
		printf("\t>First gap: %d\n", values.first);
		printf("\t>Gap extension: %d\n\n", values.extension);
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

SWquery::SWquery(FASTAsequence *first, FASTAsequence *second) {
	if(first == NULL || second == NULL)
		exitWithMsg("Input for SWquerry must not be NULL.", -1);

	this->first = first;
	this->second = second;
	prepared = false;
}

void SWquery::prepare(LaunchConfig config) {
	if(first->getLength() < second->getLength()) {
		FASTAsequence *temp = first;
		first = second;
		second = temp;
	}

    if(!first->doPaddingForRows() || !second->doPaddingForColumns(config.blocks))
    	exitWithMsg("An error has occured while applying padding on input sequences.", -1);

    deviceFirst = (char *) cudaGetDeviceCopy(
        		first->getSequence(),
        		first->getPaddedLength() * sizeof(char)
        		);

    deviceSecond = (char *) cudaGetDeviceCopy(
        		second->getSequence(),
        		second->getPaddedLength() * sizeof(char)
        		);

    prepared = true;
}

SWquery::~SWquery() {
	safeAPIcall(cudaFree(deviceFirst), __LINE__);
	safeAPIcall(cudaFree(deviceSecond), __LINE__);
}

char *SWquery::getDevFirst() {
	checkPrepared();
	return deviceFirst;
}

char *SWquery::getDevSecond() {
	checkPrepared();
	return deviceSecond;
}

FASTAsequence *SWquery::getFirst() {
	checkPrepared();
	return first;
}

FASTAsequence *SWquery::getSecond() {
	checkPrepared();
	return second;
}

void *cudaGetSpaceAndSet(int size, int setTo) {
	void *devPointer = NULL;
	safeAPIcall(cudaMalloc(&devPointer, size), __LINE__);
	safeAPIcall(cudaMemset(devPointer, setTo, size), __LINE__);
	return devPointer;
}

void *cudaGetDeviceCopy(void *src, int size) {
	void *devicePointer = cudaGetSpaceAndSet(size, 0);
	safeAPIcall(cudaMemcpy(devicePointer, src, size, cudaMemcpyHostToDevice), __LINE__);
	return devicePointer;
}

void initVerticalBuffer(VerticalBuffer *vBuffer, LaunchConfig config) {
	vBuffer->diagonal = (int *) cudaGetSpaceAndSet(
			config.blocks * config.threads * sizeof(int), 0);
    vBuffer->left0 = (int2 *) cudaGetSpaceAndSet(
    		config.blocks * config.threads * sizeof(int2), 0);
	vBuffer->left1 = (int2 *) cudaGetSpaceAndSet(
			config.blocks * config.threads * sizeof(int2), 0);
	vBuffer->left2 = (int2 *) cudaGetSpaceAndSet(
			config.blocks * config.threads * sizeof(int2), 0);
	vBuffer->left3 = (int2 *) cudaGetSpaceAndSet(
			config.blocks * config.threads * sizeof(int2), 0);
}

void freeVerticalBuffer(VerticalBuffer *vBuffer) {
	safeAPIcall(cudaFree(vBuffer->diagonal), __LINE__);
	safeAPIcall(cudaFree(vBuffer->left0), __LINE__);
	safeAPIcall(cudaFree(vBuffer->left1), __LINE__);
	safeAPIcall(cudaFree(vBuffer->left2), __LINE__);
	safeAPIcall(cudaFree(vBuffer->left3), __LINE__);
}

void initGlobalBuffer(GlobalBuffer *buffer, int secondLength, LaunchConfig config) {
	initVerticalBuffer(&buffer->vBuffer, config);
	buffer->hBuffer.up = (int2 *) cudaGetSpaceAndSet(secondLength * sizeof(int2), 0);
}

void freeGlobalBuffer(GlobalBuffer *buffer) {
	freeVerticalBuffer(&buffer->vBuffer);
    safeAPIcall(cudaFree(buffer->hBuffer.up), __LINE__);
}

TracebackScore getTracebackScore(scoring values, int row, int cols,
		int2 *vBusOut, int2 *specialRow, int targetScore, int absColIdx) {
										//dodamo target score kao argument, i kad njega nademo, to je crosspoint
	int gapOpen = -values.first;
	int gapExtend = -values.extension;

	//printf("target = %d, absColIdx = %d, row = %d, rows = %d, cols = %d\n",
		//	targetScore, absColIdx, row, rows, cols);
	printf("SR0 = %d\n", specialRow[0].x);
	TracebackScore score;

	int sRowIdx, vBusIdx;
	for(sRowIdx = 0, vBusIdx = cols - 1; sRowIdx < cols; ++sRowIdx, --vBusIdx) {

		int rScore =  specialRow[sRowIdx].x;
		int rAffine =  specialRow[sRowIdx].y;

		int bScore = vBusOut[vBusIdx].x;
		int bAffine = vBusOut[vBusIdx].y;

		int scr = rScore + bScore;
		int aff = rAffine + bAffine + gapOpen - gapExtend;
	//	printf("scr = %d, aff = %d\n", scr, aff);
		//printf("rScore = %d, bScore = %d, SRowIdx = %d\n\n", rScore, bScore, sRowIdx);

		int isScrAff = (rScore == rAffine) && (bScore == bAffine);

	/*	if(targetScore - scr < 100) {
			printf("Diff scr %d, target = %d, rS = %d, bS = %d\n", targetScore - scr, targetScore, rScore, bScore);
		}
		if(targetScore - aff < 100) {
			printf("Diff aff %d\n", targetScore - aff);
		}*/

		if (scr == targetScore && !isScrAff) {
			score.gap = 0;
			score.column = absColIdx - cols + sRowIdx;
			score.row = row;
			score.score = rScore;
			
			return score;
		}

		if (aff == targetScore) {
			score.gap = 1;
			score.column = absColIdx - cols + sRowIdx;
			score.row = row;
			score.score = rAffine;
			
			return score;
		}
	}
	
	score.column = score.row = score.score = -1;
	
	return score;
}

LaunchConfig getLaunchConfig(int shorterSeqLength, CUDAcard gpu) {
	LaunchConfig config;
	
	config.blocks = gpu.cudaCores / 2;
	config.threads = gpu.maxThreadsPerBlock / 8;

	//config.blocks = 2;
	//config.threads = 5;
	
	if (config.threads * config.blocks * 2 > shorterSeqLength)
        config.blocks = (int) ((float) (shorterSeqLength) / (config.threads * 2));

    if (config.blocks == 0) {
        config.blocks = 1;
        config.threads = shorterSeqLength / ALPHA;
    }

    config.sharedMemSize = config.threads * sizeof(int2);

    return config;
}

int _ConvertSMVer2Cores(int major, int minor) {
    typedef struct {
        int SM;
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
            return nGpuArchCoresPerSM[index].Cores;

        index++;
    }

    return -1;
}


alignmentScore getMaxScore(alignmentScore *score, int n) {
	alignmentScore max;
	max.score = -1;
	max.row = -1;
	max.column = -1;

	for(int i = 0; i < n; i++) {
		if(score[i].score > max.score)
			max = score[i];
	}

	return max;
}

CUDAcard findBestDevice() {
	int numOfDevices, bestDeviceNumber;

	cudaDeviceProp bestDeviceProps;
	
	safeAPIcall(cudaGetDeviceCount(&numOfDevices), __LINE__);
	
	int maxCores = -1;

	for (int i = 0; i < numOfDevices; ++i) {
		cudaDeviceProp currentDeviceProps;
		safeAPIcall(cudaGetDeviceProperties(&currentDeviceProps, i), __LINE__);
			
		int deviceCores = _ConvertSMVer2Cores(currentDeviceProps.major,
				currentDeviceProps.minor) * currentDeviceProps.multiProcessorCount;

		if (maxCores < deviceCores) {
			maxCores = deviceCores;
			bestDeviceNumber = i;
			bestDeviceProps = currentDeviceProps;
		}
	}

	if(maxCores < 0 || numOfDevices < 1)
		exitWithMsg("No CUDA capable card detected.", -2);
	
	CUDAcard gpu;
	gpu.cardNumber = bestDeviceNumber;
	gpu.major = bestDeviceProps.major;
	gpu.minor = bestDeviceProps.minor;
	gpu.cardsInSystem = numOfDevices;
	gpu.maxThreadsPerBlock = bestDeviceProps.maxThreadsDim[0];
	gpu.SMs = bestDeviceProps.multiProcessorCount;
	gpu.cudaCores = maxCores;
	gpu.globalMem = bestDeviceProps.totalGlobalMem;
	strcpy(gpu.name, bestDeviceProps.name);

	return gpu;
}

void printLaunchConfig(LaunchConfig config) {
	printf("Launch configuration:\n");
	printf("\t>Blocks: %d\n", config.blocks);
	printf("\t>Threads: %d\n", config.threads);
	printf("\t>Shared Memory: %d Bytes per block\n", config.sharedMemSize);
}

void printCardInfo(CUDAcard gpu) {
	printf("\t>Name: %s\n", gpu.name);
	printf("\t>CUDA Capability: %d.%d\n", gpu.major, gpu.minor);
	printf("\t>Global memory: %.0f MBytes\n", (float) gpu.globalMem / 1048576.0f);
	printf("\t>Multiprocessors: %d\n", gpu.SMs);
	printf("\t>CUDA cores: %d\n", gpu.cudaCores);
}
