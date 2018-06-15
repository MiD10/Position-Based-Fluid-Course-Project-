#ifndef PARTICLE_KERNEL_H
#define PARTICLE_KERNEL_H

#include <cuda_runtime.h>

// simulation parameters
struct SimParams
{
	float3 colliderPos;
	float  colliderRadius;

	float3 gravity;
	float globalDamping;
	float particleRadius;

	unsigned int gridSize;
	unsigned int numCells;
	float3 worldOrigin;
	float cellSize;

	unsigned int numBodies;
	unsigned int maxParticlesPerCell;

	float spring;
	float damping;
	float shear;
	float attraction;
	float boundaryDamping;
};

#endif