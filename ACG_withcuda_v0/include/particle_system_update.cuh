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
__device__ int3 calcGridPos(float3 p)
{
	int3 gridPos = make_int3(0, 0, 0);
	gridPos.x = int(p.x / params.cellSize.x) % params.gridSize.x;
	gridPos.y = int(p.y / params.cellSize.y) % params.gridSize.y;
	gridPos.z = int(p.z / params.cellSize.z) % params.gridSize.z;
	/*printf("px = %.4f, py = %.4f, pz = %.4f\n", p.x, p.y, p.z);
	printf("gx = %d, gy = %d, gz = %d\n", gridPos.x, gridPos.y, gridPos.z);*/
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ int calcGridHash(int3 gridPos)
{
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

	newPos[index] = make_float4(pos, 1.0f);
	velocity[index] = make_float4(vel, 0.0f);
	/*printf("newPOS: (%f, %f, %f, %f)\nnewVEL: (%f, %f, %f, %f)\n",
		newPos[index].x, newPos[index].y, newPos[index].z, newPos[index].w,
		velocity[index].x, velocity[index].y, velocity[index].z, velocity[index].w);*/
}

__global__
void clearCells(
	uint* cells_count
)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numCells) return;

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
	if (index >= params.numCells) return;

	int3 pos = calcGridPos(make_float3(newPos[index]));
	int cellHash = calcGridHash(pos);
	cells_count[cellHash];
	int i = atomicAdd(&cells_count[cellHash], 1);
	i = min(i, params.maxParticlesPerCell - 1);
	//cells[cellHash * params.maxParticlesPerCell + i] = index;
	//printf("cellhash = %d, i = %d, index = %d\n",cellHash,i, index);
}

__device__
bool isLegalCell(int3 h) {
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

	int3 GridPos = calcGridPos(make_float3(newPos[index]));
	int neighborIndex;
	for (int z = -1; z <= 1; z++) {
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				int3 neighborCellPos = make_int3(GridPos.x + x, GridPos.y + y, GridPos.z + z);
				if (isLegalCell(neighborCellPos)) {
					int neighborCellHash = calcGridHash(neighborCellPos);
					int cellParticleNum = min(cells_count[neighborCellHash], params.maxParticlesPerCell - 1);
					int CellOffset = neighborCellHash * params.maxParticlesPerCell;
					int ParticleOffset = index * params.maxNeighborsPerParticle;
					for (int i = 0; i < cellParticleNum; i++) {
						if (neighbors_count[index] >= params.maxNeighborsPerParticle) return;
						
						neighborIndex = cells[CellOffset + i];
						if (neighborIndex == index)
							continue;
						if (length(make_float3(newPos[index]) - make_float3(newPos[neighborIndex])) <= params.kernelRadius) {
							neighbors[ParticleOffset + neighbors_count[index]] = neighborIndex;
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
		uint offset = index * params.maxNeighborsPerParticle;
		for (int i = 0; i < neighbors_count[index]; i++) {
			_density += Wpoly6(make_float3(newPos[index]), make_float3(newPos[neighbors[offset + i]]));
		}
	}
	density[index] = _density / params.restDensity - 1;
	//printf("indes: %d, density: %f\n_density: %f\n",index, density[index], _density);
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
		uint offset = index * params.maxNeighborsPerParticle;
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
	float4* newPos,
	float4* deltaPos,
	uint*	neighbors,
	uint*	neighbors_count,
	float*	lamda
)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= params.numBodies) return;

	float3 dP = make_float3(0.f);
	if (neighbors_count[index]) {
		int offset = index * params.maxNeighborsPerParticle;
		for (int i = 0; i < neighbors_count[index]; i++) {
			float3 temp = Wspiky(make_float3(newPos[index]), make_float3(newPos[neighbors[offset + i]]));
			dP += (lamda[index] + lamda[neighbors[offset + i]]) * temp;
		}
	}
	deltaPos[index] = make_float4(dP / params.restDensity, 0.f);
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