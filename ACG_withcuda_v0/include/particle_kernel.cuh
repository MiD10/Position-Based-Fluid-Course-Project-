#ifndef PARTICLE_KERNEL_H
#define PARTICLE_KERNEL_H

#include <cuda_runtime.h>

// simulation parameters
struct SimParams
{
	float deltaTime;

	float3 colliderPos;
	float  colliderRadius;

	float3 gravity;
	float globalDamping;
	float particleRadius;

	int3 gridSize;
	unsigned int number_grid_cells;
	float3 cellSize;
	float3 worldBounds;

	unsigned int numBodies;
	unsigned int maxParticlesPerCell;

	float spring;
	float damping;
	float shear;
	float attraction;
	float boundaryDamping;
};

#endif