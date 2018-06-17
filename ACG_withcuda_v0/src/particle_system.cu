#include "../include/GL/glew.h"
#include "../include/GLFW/glfw3.h"

#ifdef _WIN32
#ifndef _WINDOWS_HAS_INCLUDED
#define _WINDOWS_HAS_INCLUDED
#include <windows.h>
#endif
#endif

#include <cstdlib>
#include <cstdio>
#include <string.h>

#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

//for checkCudaErrors() function
#include <driver_types.h>	
#include "../include/helper_cuda.h"

#include "../include/particle_kernel.cuh"
#include "../include/particle_system_update.cuh"


extern "C" {

	// compute grid and thread block size for a given number of elements
	void computeGridSize(unsigned int n, unsigned int blockSize, unsigned int &numBlocks, unsigned int &numThreads)
	{
		numThreads = (blockSize < n) ? blockSize : n;
		numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
		return;
	}

	void update_fluid(
		float		  *Vel,
		float		  *oldPos,
		float		  *newPos,
		float		  *particleDensity,
		float		  *particleLamda,
		float		  *particleDeltaPos,
		unsigned int  *neighbors,
		unsigned int  *neighbors_count,
		unsigned int  *cells,
		unsigned int  *cells_count,
		unsigned int  numParticles,
		unsigned int  numGrids,
		unsigned int  iters
	)
	{
		// thread per particle
		/*unsigned int numThreads, numBlocks;
		unsigned int grid_numThreads, grid_numBlocks;
		computeGridSize(numParticles, 128, numBlocks, numThreads);
		computeGridSize(numGrids, 128, grid_numBlocks, grid_numThreads);*/
		int numThreads = 128;
		int grid_numBlocks = int(ceil(numGrids / numThreads + 0.5f));
		int numBlocks = int(ceil(numParticles / numThreads + 0.5f));

		//update position(using velocity and bound)
		updatePositionD <<< numBlocks, numThreads >>> (
			(float4*)oldPos,
			(float4*)newPos,
			(float4*)Vel
			);

		getLastCudaError("Kernel execution failed");
		clearCells <<< grid_numBlocks, numThreads >>> (
			cells_count
			);

		getLastCudaError("Kernel execution failed");
		clearNeighbors <<< numBlocks, numThreads >>> (
			neighbors_count
			);

		getLastCudaError("Kernel execution failed");
		updateCells <<< numBlocks, numThreads >>> (
			(float4*)newPos,
			cells,
			cells_count
			);

		getLastCudaError("Kernel execution failed");
		updateNeighbors <<< numBlocks, numThreads >>> (
			(float4*)newPos,
			cells,
			cells_count,
			neighbors,
			neighbors_count
			);

		getLastCudaError("Kernel execution failed");
		for (unsigned int i = 0; i < iters; i++) {
			// get each particle's C(density)
			getDensityD <<< numBlocks, numThreads >>>(
				(float4 *)newPos,
				neighbors,
				neighbors_count,
				particleDensity
				);

			getLastCudaError("Kernel execution failed");
			// get each particle's Lamda
			getLamdaD <<< numBlocks, numThreads >>> (
				(float4 *)newPos,
				neighbors,
				neighbors_count,
				particleDensity,
				particleLamda
				);

			getLastCudaError("Kernel execution failed");
			// get each particle's fixed position delta-p
			getDpD <<< numBlocks, numThreads >>>(
				(float4 *)newPos,
				(float4 *)particleDeltaPos,
				neighbors,
				neighbors_count,
				particleLamda
				);

			getLastCudaError("Kernel execution failed");
			updatePositionD <<< numBlocks, numThreads >>>(
				(float4 *)particleDeltaPos,
				(float4 *)newPos
				);
		}

		getLastCudaError("Kernel execution failed");
		updateVelocity <<< numBlocks, numThreads >>> (
			(float4 *)oldPos,
			(float4 *)newPos,
			(float4 *)Vel
			);
		//check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");
	}

	void cudaInit(int argc, char **argv)
	{
		int devID;

		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
		devID = findCudaDevice(argc, (const char **)argv);

		if (devID < 0)
		{
			printf("No CUDA Capable devices found, exiting...\n");
			exit(EXIT_SUCCESS);
		}
	}

	void allocateArray(void **devPtr, int size) 
	{
		checkCudaErrors(cudaMalloc(devPtr, size));
	}

	void freeArray(void *devPtr)
	{
		checkCudaErrors(cudaFree(devPtr));
	}

	void copyArrayToDevice(void *device, const void *host, int offset, int size)
	{
		checkCudaErrors(cudaMemcpy((char *)device + offset, host, size, cudaMemcpyHostToDevice));
	}

	void registerGLBufferObject(unsigned int vbo, struct cudaGraphicsResource **cuda_vbo_resource)
	{
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
			cudaGraphicsMapFlagsNone));
	}

	void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
	{
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
	}

	void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
	{
		void *ptr;
		checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
			*cuda_vbo_resource));
		return ptr;
	}

	void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
	{
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
	}

	void copyArrayFromDevice(void *host, const void *device,
							 struct cudaGraphicsResource **cuda_vbo_resource, int size)
	{
		if (cuda_vbo_resource)
		{
			device = mapGLBufferObject(cuda_vbo_resource);
		}

		checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

		if (cuda_vbo_resource)
		{
			unmapGLBufferObject(*cuda_vbo_resource);
		}
	}

	void setParameters(SimParams *hostParams)
	{
		// copy parameters to constant memory
		checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
	}
}