#include "../include/particle_system.h"

extern "C"
{
#ifndef PARTICLE_SYSTEM_CUH_
#define PARTICLE_SYSTEM_CUH_

	typedef unsigned int uint;

	void cudaInit(int argc, char **argv);

	void allocateArray(void **devPtr, int size);
	void freeArray(void *devPtr);

	void threadSync();

	void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
	void copyArrayToDevice(void *device, const void *host, int offset, int size);
	void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
	void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
	void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
	void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);


	void setParameters(SimParams *hostParams);

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
			unsigned int  *cell_count,
			unsigned int  numParticles,
			unsigned int  iters
	);
#endif
}