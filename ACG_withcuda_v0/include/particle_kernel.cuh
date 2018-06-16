#ifndef PARTICLE_KERNEL_H
#define PARTICLE_KERNEL_H

#include <cuda_runtime.h>

// simulation parameters
struct SimParams
{
	float3 colliderPos;
	float  colliderRadius;

	float deltaTime;
	float3 gravity;
	float globalDamping;
	float kernelRadius;
	float particleRadius;

	int3 gridSize;
	unsigned int numCells;
	float3 worldbound;
	float3 cellSize;

	unsigned int numBodies;
	unsigned int maxParticlesPerCell;
	unsigned int maxNeighborsPerParticle;

	float spring;
	float damping;
	float shear;
	float attraction;
	float boundaryDamping;

	//pbf
	unsigned int numIterations;
	float restDensity;
	float relaxation;
	float poly6;
	float spiky;
};

#endif