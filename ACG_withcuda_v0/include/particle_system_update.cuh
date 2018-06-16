#ifndef PARTICLE_SYSTEM_CUH
#define PARTICLE_SYSTEM_CUH


#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;


#ifdef _WIN32
#ifndef _WINDOWS_HAS_INCLUDED
#define _WINDOWS_HAS_INCLUDED
#include <windows.h>
#endif
#endif



#include "../include/particle_kernel.cuh"

#include "../include/helper_math.h"

typedef unsigned int uint;

// simulation parameters in constant memory
__constant__ SimParams params;

// calculate position in uniform grid
__device__ uint3 calcGridPos(float3 p)
{
	uint3 gridPos;
	gridPos.x = floor((p.x) / params.cellSize.x);
	gridPos.y = floor((p.y) / params.cellSize.y);
	gridPos.z = floor((p.z) / params.cellSize.z);
	/*printf("px = %.4f, py = %.4f, pz = %.4f\n", p.x, p.y, p.z);
	printf("gx = %d, gy = %d, gz = %d\n", gridPos.x, gridPos.y, gridPos.z);*/
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(uint3 gridPos)
{
	gridPos.x = gridPos.x % params.gridSize.x;  
	gridPos.y = gridPos.y % params.gridSize.y;
	gridPos.z = gridPos.z % params.gridSize.z;
	return gridPos.z * params.gridSize.x * params.gridSize.y + gridPos.y * params.gridSize.x + gridPos.x;
}

__device__
float Wpoly6(
	float3 i,
	float3 j
)
{
	float3 r = i - j;
	float len = length(r);
	if (len > params.kernelRadius) {
		return 0;
	}
	else {
		return params.poly6 * pow((params.kernelRadius * params.kernelRadius - len * len), 3);
	}
}

__device__
float3 Wspiky(
	float3 i,
	float3 j
)
{
	float3 r = i - j;
	float len = length(r);
	if (len > params.kernelRadius) {
		return make_float3(0.0f);
	}
	else {
		return params.spiky * pow((params.kernelRadius - len), 2) * (r / len);
	}
}

__global__
void updatePositionD(
	float4* oldPos,
	float4* newPos,
	float4* velocity
) 
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBodies) return;

	float3 pos = make_float3(oldPos[index]);
	float3 vel = make_float3(velocity[index]);

	vel += params.gravity * params.deltaTime;
	//vel *= params.globalDamping;

	// new position = old position + velocity * deltaTime
	pos += vel * params.deltaTime;

	if (pos.x < 0) {
		vel.x = 0;
		pos.x = 0.001f;
	}
	else if (pos.x > params.worldbound.x) {
		vel.x = 0;
		pos.x = params.worldbound.x - 0.001f;
	}

	if (pos.y < 0) {
		vel.y = 0;
		pos.y = 0.001f;
	}
	else if (pos.y > params.worldbound.y) {
		vel.y = 0;
		pos.y = params.worldbound.y - 0.001f;
	}

	if (pos.z < 0) {
		vel.z = 0;
		pos.z = 0.001f;
	}
	else if (pos.z > params.worldbound.z) {
		vel.z = 0;
		pos.z = params.worldbound.z - 0.001f;
	}

	newPos[index] = make_float4(pos, 0.0f);
	velocity[index] = make_float4(vel, 0.0f);
}

__global__
void clearCells(
	uint* cells_count
)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBodies) return;

	cells_count[index] = 0;
}

__global__
void clearNeighbors(
	uint* neighbors_count
)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBodies) return;

	neighbors_count[index] = 0;
}

__global__
void updateCells(
	float4* newPos,
	uint*	cells,
	uint*	cells_count
) 
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBodies) return;

	uint3 pos = calcGridPos(make_float3(newPos[index]));
	uint cellHash = calcGridHash(pos);

	uint i = atomicAdd(&cells_count[cellHash], 1);
	i = min(i, params.maxParticlesPerCell);
	cells[cellHash * params.maxParticlesPerCell + i] = index;
}

__device__
bool isLegalCell(uint3 h) {
	if (h.x >= 0 && h.x < params.gridSize.x && h.y >= 0 && h.y < params.gridSize.y && h.z >= 0 && h.z < params.gridSize.z)
		return true;
	else
		return false;

}
__global__
void updateNeighbors(
	float4* newPos,
	uint*	cells,
	uint*	cells_count,
	uint*	neighbors,
	uint*	neighbors_count
)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBodies) return;

	uint3 pos = calcGridPos(make_float3(newPos[index]));
	uint neighborIndex;
	for (int z = -1; z <= 1; z++) {
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				uint3 neighborCellPos = make_uint3(pos.x + x, pos.y + y, pos.z + z);
				if (isLegalCell(neighborCellPos)) {
					uint neighborCellHash = calcGridHash(neighborCellPos);
					uint cellParticleNum = min(cells_count[neighborCellHash], params.maxParticlesPerCell - 1);
					uint offset = neighborCellHash * params.maxParticlesPerCell;
					for (int i = 0; i < cellParticleNum; i++) {
						if (neighbors_count[index] >= params.maxNeighborsPerParticle) return;
						
						neighborIndex = cells[offset + i];
						if (length(make_float3(newPos[index]) - make_float3(newPos[neighborIndex])) <= params.kernelRadius) {
							neighbors[offset + neighbors_count[index]] = neighborIndex;	//just use the same offset here
																						//may change, may not
							neighbors_count[index]++;
						}
					}
				}
			}
		}
	}
}

__global__
void getDensityD(
	float4* newPos,
	uint*	neighbors,
	uint*	neighbors_count,
	float*	density
) 
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBodies) return;

	float _density = 0.f;
	if (neighbors_count[index]) {
		uint offset = index + params.maxNeighborsPerParticle;
		for (int i = 0; i < neighbors_count[index]; i++) {
			_density += Wpoly6(make_float3(newPos[index]), make_float3(newPos[neighbors[offset + i]]));
		}
	}
	density[index] = _density / params.restDensity - 1;
}

__global__
void getLamdaD(
	float4* newPos,
	uint*	neighbors,
	uint*	neighbors_count,
	float*	density,
	float*	lamda
)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBodies) return;

	float3 gradSelf = make_float3(0.f);
	float grad = 0.f;
	if (neighbors_count[index]) {
		uint offset = index + params.maxNeighborsPerParticle;
		for (int i = 0; i < neighbors_count[index]; i++) {
			float3 gradNeighbor = Wspiky(make_float3(newPos[index]), make_float3(newPos[neighbors[offset + i]]));
			gradSelf += gradNeighbor;
			float len = length(gradNeighbor);
			grad += len * len;
		}
	}
	float len = length(gradSelf);
	grad += len * len;
	lamda[index] = (-1 * density[index]) / (grad + params.relaxation);
}

__global__
void getDpD(
	float4* deltaPos,
	float4* newPos,
	uint*	neighbors,
	uint*	neighbors_count,
	float*	lamda
)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBodies) return;

	deltaPos[index] = make_float4(0.f);
	float3 dP = make_float3(0.f);
	if (neighbors_count[index]) {
		uint offset = index + params.maxNeighborsPerParticle;
		for (int i = 0; i < neighbors_count[index]; i++) {
			float3 temp = Wspiky(make_float3(newPos[index]), make_float3(newPos[neighbors[offset + i]]));
			dP += (lamda[index] + lamda[offset + i]) * temp / params.restDensity;
		}
	}
	deltaPos[index] = make_float4(dP, 0.f);
}

__global__
void updatePositionD(
	float4* deltaPos,
	float4* newPos
)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBodies) return;

	newPos[index] += deltaPos[index];
}

__global__
void updateVelocity(
	float4* oldPos,
	float4* newPos,
	float4* vel
)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBodies) return;

	vel[index] = make_float4((make_float3(newPos[index]) - make_float3(oldPos[index])) / params.deltaTime, 0.f);

	oldPos[index] = newPos[index];
}

#endif