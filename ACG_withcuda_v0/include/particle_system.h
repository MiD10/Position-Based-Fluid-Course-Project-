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
	ParticleSystem(SimParams& temp);

	unsigned int number;
	unsigned int number_grid_cells;
	/*vector<Partical> particles;*/
	unsigned int posVAO;
	unsigned int colorVAO;

	//CPU data
	float* host_Position;	//position array in cpu memory
	float* host_Velocity;	//velocity array in cpu memory
	float* host_force;		//force array in cpu memory

	//GPU data
	float* device_Position; //position array in gpu memory
	float* device_Velocity;	//velocity array in gpu memory
	float* device_force;	//force array in gpu memory

	//uniformed grid
	float* sorted_device_Position;
	float* sorted_device_Velocity;
	float* sorted_device_force;
	unsigned int* grid_cell_start;		//start of each sorted cell
	unsigned int* grid_cell_end;		//end of each sorted cell

	unsigned int* device_grid_particle_hash;	//particle hash for each particle
	unsigned int* device_grid_particle_index;	//particle index for each particle
	unsigned int* device_grid_cell_start;		//start of each sorted cell
	unsigned int* device_grid_cell_end;			//end of each sorted cell

	unsigned int gird_sort_bits;

	float *cudaPosVBO;        // these are the CUDA deviceMem Pos
	float *cudaColorVBO;      // these are the CUDA deviceMem Color

	
	//point to the same thing in opengl and in CUDA
	unsigned int posVBO;         // vertex buffer object for particle positions
	unsigned int colorVBO;       // vertex buffer object for colors
	struct cudaGraphicsResource* cuda_posvbo_resource; // handles OpenGL-CUDA exchange
	struct cudaGraphicsResource* cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

	//params
	SimParams params;

	//important functions
	void initialize(void);
	
	void resetRandom(void); //reset all the particles's position and velocity randomly
	void resetGrid(void); //
	void update(void);

	void createPosVBO(unsigned int size);
	//void createColorVBO(unsigned int size);

	void draw(MyShader& omyShader);

	//debugging
	void dumpParticles(unsigned int start, unsigned int count);
};

#endif
