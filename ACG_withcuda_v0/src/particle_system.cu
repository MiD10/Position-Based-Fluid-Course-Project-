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

	void integrateSystem(
		float *pos,
		float *vel,
		float deltaTime,
		unsigned int numParticles)
	{
		thrust::device_ptr<float4> d_pos4((float4 *)pos);
		thrust::device_ptr<float4> d_vel4((float4 *)vel);

		thrust::for_each(
			thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4)),
			thrust::make_zip_iterator(thrust::make_tuple(d_pos4 + numParticles, d_vel4 + numParticles)),
			integrate_functor(deltaTime));
	}

	// compute grid and thread block size for a given number of elements
	void computeGridSize(unsigned int n, unsigned int blockSize, unsigned int &numBlocks, unsigned int &numThreads)
	{
		numThreads = (blockSize < n) ? blockSize : n;
		numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
		return;
	}

	void calcHash(
		unsigned int  *gridParticleHash,
		unsigned int  *gridParticleIndex,
		float *pos,
		int    numParticles)
	{
		unsigned int numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		// execute the kernel
		calcHashD <<< numBlocks, numThreads >>>(gridParticleHash,
			gridParticleIndex,
			(float4 *)pos,
			numParticles);

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");
	}

	void reorderDataAndFindCellStart(
		unsigned int  *cellStart,
		unsigned int  *cellEnd,
		float *sortedPos,
		float *sortedVel,
		unsigned int  *gridParticleHash,
		unsigned int  *gridParticleIndex,
		float *oldPos,
		float *oldVel,
		unsigned int   numParticles,
		unsigned int   numCells) 
	{
		unsigned int numThreads, numBlocks;
		computeGridSize(numParticles, 256, numBlocks, numThreads);

		// set all cells to empty
		checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells * sizeof(unsigned int)));

#if USE_TEX
		checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles * sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldVelTex, oldVel, numParticles * sizeof(float4)));
#endif
		unsigned int smemSize = sizeof(unsigned int)*(numThreads + 1);
		reorderDataAndFindCellStartD <<< numBlocks, numThreads, smemSize >>>(
			cellStart,
			cellEnd,
			(float4 *)sortedPos,
			(float4 *)sortedVel,
			gridParticleHash,
			gridParticleIndex,
			(float4 *)oldPos,
			(float4 *)oldVel,
			numParticles);
		getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

#if USE_TEX
		checkCudaErrors(cudaUnbindTexture(oldPosTex));
		checkCudaErrors(cudaUnbindTexture(oldVelTex));
#endif
	}

	void collide(
		float *newVel,
		float *sortedPos,
		float *sortedVel,
		unsigned int  *gridParticleIndex,
		unsigned int  *cellStart,
		unsigned int  *cellEnd,
		unsigned int   numParticles,
		unsigned int   numCells)
	{
#if USE_TEX
		checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles * sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles * sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells * sizeof(unsigned int)));
		checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells * sizeof(unsigned int)));
#endif

		// thread per particle
		unsigned int numThreads, numBlocks;
		computeGridSize(numParticles, 64, numBlocks, numThreads);

		// execute the kernel
		collideD <<< numBlocks, numThreads >>>(
			(float4 *)newVel,
			(float4 *)sortedPos,
			(float4 *)sortedVel,
			gridParticleIndex,
			cellStart,
			cellEnd,
			numParticles);

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");

#if USE_TEX
		checkCudaErrors(cudaUnbindTexture(oldPosTex));
		checkCudaErrors(cudaUnbindTexture(oldVelTex));
		checkCudaErrors(cudaUnbindTexture(cellStartTex));
		checkCudaErrors(cudaUnbindTexture(cellEndTex));
#endif
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

	void sortParticles(unsigned int *dGridParticleHash, unsigned int *dGridParticleIndex, unsigned int numParticles)
	{
		thrust::sort_by_key(
			thrust::device_ptr<unsigned int>(dGridParticleHash),
			thrust::device_ptr<unsigned int>(dGridParticleHash + numParticles),
			thrust::device_ptr<unsigned int>(dGridParticleIndex));
	}
}