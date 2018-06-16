#ifndef PARTICLE_SYSTEM_H
#define PARTICLE_SYSTEM_H

#ifdef _WIN32
#ifndef _WINDOWS_HAS_INCLUDED
#define _WINDOWS_HAS_INCLUDED
#include <windows.h>
#endif
#endif

//cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include "../include/particle_kernel.cuh"


#include "../include/myshader.h"
#include "../include/mycamera.h"

using namespace std;

class ParticleSystem {
public:
	ParticleSystem(float dT, unsigned int number_of_particles, int3 gridSize);

	unsigned int number;
	unsigned int number_grid_cells;
	/*vector<Partical> particles;*/
	unsigned int posVAO;
	unsigned int colorVAO;

	//CPU data: for debug only
	float* host_Position;			//position array in cpu memory
	float* host_Velocity;			//velocity array in cpu memory
	float* host_force;				//force array in cpu memory
	float* host_density;
	float* host_lamda;
	float* host_delta_Position;
	unsigned int* host_neighborsCount;
	unsigned int* host_neighbors;
	unsigned int* host_cells_count;
	unsigned int* host_cells;


	//GPU data
	float* device_Position;			//position array in gpu memory
	float* device_Velocity;			//velocity array in gpu memory
	float* device_force;			//force array in gpu memory

	//uniformed grid
	unsigned int* device_neighbors;				//neighbors of one particle
	unsigned int* device_neighbors_count;		//number of neighbors
	unsigned int* device_cells;					//particles in the cell
	unsigned int* device_cells_count;			//number of particles

	//sloving pbf
	float* device_new_Position;		//store the new position calculated in solving iterations
	float* device_density;			//store the density of each particle
	float* device_lamda;			//store the lamda of each particle
	float* device_delta_Position;	//store the calculated delta position

	unsigned int gird_sort_bits;

	float *cudaPosVBO;				// these are the CUDA deviceMem Pos
	float *cudaColorVBO;			// these are the CUDA deviceMem Color

	
	//point to the same thing in opengl and in CUDA
	unsigned int posVBO;									// vertex buffer object for particle positions
	unsigned int colorVBO;									// vertex buffer object for colors
	struct cudaGraphicsResource* cuda_posvbo_resource;		// handles OpenGL-CUDA exchange
	struct cudaGraphicsResource* cuda_colorvbo_resource;	// handles OpenGL-CUDA exchange

	//params
	SimParams params;

	//important functions
	void initialize(void);
	
	void resetRandom(void); //reset all the particles's position and velocity randomly
	void resetGrid(void); //
	void update();

	void createPosVBO(unsigned int size);
	//void createColorVBO(unsigned int size);

	void draw(MyShader& omyShader);

	//debugging
	void dumpParticles(unsigned int start, unsigned int count);
	void dumpDensity_Lamda();
	void dumpLamda();
	void dumpDeltaPosition();
	void dumpNeighbors();
	void dumpCells();
};

#endif
