#include <cuda.h>
#include <stdio.h>

int main(int argc, char **argv) {
	int devCount;
	cudaGetDeviceCount(&devCount);

	printf("Number of CUDA capable devices: %d\n\n", devCount);

	cudaDeviceProp propreties;
	for(int i = 0; i < devCount; i++) {
		cudaGetDeviceProperties(&propreties, i);

		printf("Device number %d\n", i + 1);
		printf("Name: %s\n", propreties.name);
		printf("Copy overlap: %s\n", propreties.deviceOverlap ? "Enabled" : "Disabled");
		printf("Total global memory: %d bytes\n", (int) propreties.totalGlobalMem);
		printf("Multiprocessor count: %d\n", propreties.multiProcessorCount);

		printf("\n");
	}

	memset(&propreties, 0, sizeof(cudaDeviceProp));

	propreties.major = 1;
	propreties.minor = 3;

	int device;
	cudaChooseDevice(&device, &propreties);
	cudaSetDevice(device);

	printf("Device that best matches given properties is device number %d\n", device + 1);
}
