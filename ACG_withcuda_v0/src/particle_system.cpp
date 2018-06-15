#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif

#include <math.h>

#include "../include/GL/glew.h"
#include "../include/GLFW/glfw3.h"

#include "../include/particle_system.h"
#include "../include/particle_system.cuh"
#include "../include/particle_kernel.cuh"

using namespace std;

//ParticleSystem::ParticleSystem(int num):number(num){
//	for (float x = 0; x < 50; x += 1)
//		for (float y = 0; y < 5; y += 1)
//			for (float z = 0; z < 50; z += 1) {
//				Partical temp(glm::vec3(x, y, z), glm::vec3(0, 0, z), glm::vec3(0, -9.8, 0));
//				particles.push_back(temp);
//			}
//	glGenVertexArrays(1, &VAO);
//	glGenBuffers(1, &VBO);
//	glBindVertexArray(VAO);
//
//	glBindBuffer(GL_ARRAY_BUFFER, VBO);
//	glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(Partical), &particles[0], GL_STATIC_DRAW);
//
//	//position
//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)0);
//	glEnableVertexAttribArray(0);
//	//velocity
//	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 10 * sizeof(float), (void*)offsetof(Partical, velocity));
//	glEnableVertexAttribArray(1);
//}


//=========================================================================
//helper funcitons:========================================================
//t = [0,1], r is the colored pointer
void colorRamp(float t, float *r) {
	const int ncolors = 7;
	float c[ncolors][3] =
	{
	{ 1.0, 0.0, 0.0, },
	{ 1.0, 0.5, 0.0, },
	{ 1.0, 1.0, 0.0, },
	{ 0.0, 1.0, 0.0, },
	{ 0.0, 1.0, 1.0, },
	{ 0.0, 0.0, 1.0, },
	{ 1.0, 0.0, 1.0, },
	};
	t = t * (ncolors - 1);
	int i = (int)t;
	float u = t - floor(t);
	//a + t * (b - a)
	r[0] = c[i][0] + (c[i + 1][0] - c[i][0]) *  u;
	r[1] = c[i][1] + (c[i + 1][1] - c[i][1]) *  u;
	r[2] = c[i][2] + (c[i + 1][2] - c[i][2]) *  u;
}

void ParticleSystem::createPosVBO(unsigned int size) {
	glGenVertexArrays(1, &posVAO);
	glGenBuffers(1, &posVBO);
	glBindVertexArray(posVAO);
	glBindBuffer(GL_ARRAY_BUFFER, posVBO);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	return;
}

unsigned int createVBO(unsigned int size)
{
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vbo;
}

//========================================================================
//class funcitons:========================================================
ParticleSystem::ParticleSystem(unsigned int number_of_particles, unsigned int gridSize){
	//param setting
	number = number_of_particles;
	params.numBodies = number;
	params.worldOrigin = make_float3(-6.4f, -6.4f, -6.4f);

	host_force = host_Position = host_Velocity = NULL;
	device_force = device_Velocity = NULL;
	params.boundaryDamping = -0.5f;	//new_velocity = velocity * boundaryDamping when bouncing to the wall|floor
	params.gravity = make_float3(0.0f, -9.8f, 0.0f);
	params.globalDamping = 1.0f; //everytime update(), new_velocity = velocity * globalDamping
	params.spring = 0.5f;
	params.damping = 0.02f;
	params.shear = 0.1f;
	params.attraction = 0.0f;
	params.particleRadius = 0.1f; //particle radius

	//collision
	//params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
	//params.colliderRadius = 0.2f;

	//grids&cells
	number_grid_cells = gridSize * gridSize * gridSize;

	params.gridSize = gridSize;
	params.numCells = number_grid_cells;
	params.cellSize = params.particleRadius * 2.0f;


	
	//(memory) initialize
	initialize();
}

void ParticleSystem::initialize() {

	//CPU memory allocation for pos/vel/force data
	host_force = new float[number * 4];
	host_Position = new float[number * 4];
	host_Velocity = new float[number * 4];
	memset(host_force, 0, number * 4 * sizeof(float));
	memset(host_Position, 0, number * 4 * sizeof(float));
	memset(host_Velocity, 0, number * 4 * sizeof(float));

	//grids
	grid_cell_start = new unsigned int[number_grid_cells];
	memset(grid_cell_start, 0, number_grid_cells * sizeof(unsigned int));
	grid_cell_end = new unsigned int[number_grid_cells];
	memset(grid_cell_end, 0, number_grid_cells * sizeof(unsigned int));

	allocateArray((void **)&device_grid_particle_hash, number * sizeof(unsigned int));
	allocateArray((void **)&device_grid_particle_index, number * sizeof(unsigned int));
	allocateArray((void **)&device_grid_cell_start, number_grid_cells * sizeof(unsigned int));
	allocateArray((void **)&device_grid_cell_end, number_grid_cells * sizeof(unsigned int));
	
	//GPU allocate
	unsigned int memSize = sizeof(float) * 4 * number;

	allocateArray((void **)&sorted_device_Position, memSize);
	allocateArray((void **)&sorted_device_Velocity, memSize);

	allocateArray((void **)&device_Velocity, memSize);

	//VBO creation and bind to cudaGraphicsResource
	createPosVBO(memSize);
	registerGLBufferObject(posVBO, &cuda_posvbo_resource);

	colorVBO = createVBO(memSize);
	registerGLBufferObject(colorVBO, &cuda_colorvbo_resource);

	// fill color buffer
	glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
	float *data = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float *ptr = data;
	for (unsigned int i = 0; i < number; i++)
	{
		float t = i / (float)number;
#if 0 //compile decision
		* ptr++ = rand() / (float)RAND_MAX;
		*ptr++ = rand() / (float)RAND_MAX;
		*ptr++ = rand() / (float)RAND_MAX;
#else
		colorRamp(t, ptr);
		ptr += 3;
#endif
		*ptr++ = 1.0f;
}
	glUnmapBuffer(GL_ARRAY_BUFFER);

	//params
	setParameters(&params);
}

void ParticleSystem::resetRandom(void) { //first edition, alllll random
	std::cout << "Particle resetting..." << std::endl;
	int p = 0, v = 0;
	for (int i = 0; i < number; i++) {
		host_Position[p++] = 2 * (rand() / (float)RAND_MAX - 0.5);
		host_Position[p++] = 2 * (rand() / (float)RAND_MAX - 0.5);
		host_Position[p++] = 2 * (rand() / (float)RAND_MAX - 0.5);
		host_Position[p++] = 1.0f;
		host_Velocity[v++] = 2 * (rand() / (float)RAND_MAX - 0.5);
		host_Velocity[v++] = 2 * (rand() / (float)RAND_MAX - 0.5);
		host_Velocity[v++] = 2 * (rand() / (float)RAND_MAX - 0.5);
		host_Velocity[v++] = 0.0f;
	}
	//register position
	unregisterGLBufferObject(cuda_posvbo_resource);
	glBindBuffer(GL_ARRAY_BUFFER, posVBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, number * 4 * sizeof(float), host_Position);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	registerGLBufferObject(posVBO, &cuda_posvbo_resource);

	//copy velocity to GPU
	copyArrayToDevice(device_Velocity, host_Velocity, 0, number * 4 * sizeof(float));
	std::cout << "Particle reset done!" << std::endl;

	return;
}

void ParticleSystem::resetGrid() {
	std::cout << "Particle resetting..." << std::endl;
	srand(1973);

	int i = 0;
	unsigned int size = (int)ceilf(powf((float)params.numBodies, 1.0f / 3.0f));
	//unsigned int size = params.gridSize;
	float spacing = params.particleRadius * 2.0f;
	float jitter = params.particleRadius * 0.01f;
	for (unsigned int z = 0; z < size; z++) {
		for (unsigned int y = 0; y < size; y++) {
			for (unsigned int x = 0; x < size; x++, i++) {
				if (i < params.numBodies) {
					//printf("%d, %d, %d, %d\n", i, x, y, z);
					host_Position[i * 4] = (spacing * x) + params.particleRadius - 1.0f + 2 * (rand() / (float)RAND_MAX - 0.5) * jitter;
					host_Position[i * 4 + 1] = (spacing * y) + params.particleRadius - 1.0f + 2 * (rand() / (float)RAND_MAX - 0.5) * jitter;
					host_Position[i * 4 + 2] = (spacing * z) + params.particleRadius - 1.0f + 2 * (rand() / (float)RAND_MAX - 0.5) * jitter;
					host_Position[i * 4 + 3] = 1.0f;

					host_Velocity[i * 4] = 0.0f;
					host_Velocity[i * 4 + 1] = 0.0f;
					host_Velocity[i * 4 + 2] = 0.0f;
					host_Velocity[i * 4 + 3] = 0.0f;
					//printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", host_Position[i * 4 + 0], host_Position[i * 4 + 1], host_Position[i * 4 + 2], host_Position[i * 4 + 3]);
					//printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", host_Velocity[i * 4 + 0], host_Velocity[i * 4 + 1], host_Velocity[i * 4 + 2], host_Velocity[i * 4 + 3]);
				}
				else {
					//register position
					unregisterGLBufferObject(cuda_posvbo_resource);
					glBindBuffer(GL_ARRAY_BUFFER, posVBO);
					glBufferSubData(GL_ARRAY_BUFFER, 0, number * 4 * sizeof(float), host_Position);
					glBindBuffer(GL_ARRAY_BUFFER, 0);
					registerGLBufferObject(posVBO, &cuda_posvbo_resource);
					//copy velocity to GPU
					copyArrayToDevice(device_Velocity, host_Velocity, 0, number * 4 * sizeof(float));
					std::cout << "Particle reset done!" << std::endl;
					return;
				}
			}
		}
	}
	//register position
	unregisterGLBufferObject(cuda_posvbo_resource);
	glBindBuffer(GL_ARRAY_BUFFER, posVBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, number * 4 * sizeof(float), host_Position);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	registerGLBufferObject(posVBO, &cuda_posvbo_resource);
	//copy velocity to GPU
	copyArrayToDevice(device_Velocity, host_Velocity, 0, number * 4 * sizeof(float));
	std::cout << "Particle reset done!" << std::endl;
	return;
}

int cccc = 0;
void ParticleSystem::update(float deltaTime) {

	device_Position = (float*)mapGLBufferObject(&cuda_posvbo_resource);
	
	setParameters(&params);

	//calculate position and new velocity using GPU
	// integrate
	integrateSystem(device_Position, device_Velocity, deltaTime, number);

	//every time calculate grid hash from scratch
	calcHash(device_grid_particle_hash, device_grid_particle_index, device_Position, number);

	// sort particles based on hash
	sortParticles(device_grid_particle_hash, device_grid_particle_index, number);

	// reorder particle arrays into sorted order and
	// find start and end of each cell
	reorderDataAndFindCellStart(
		device_grid_cell_start,
		device_grid_cell_end,
		sorted_device_Position,
		sorted_device_Velocity,
		device_grid_particle_hash,
		device_grid_particle_index,
		device_Position,
		device_Velocity,
		number,
		number_grid_cells);

	// process collisions
	collide(
		device_Velocity,
		sorted_device_Position,
		sorted_device_Velocity,
		device_grid_particle_index,
		device_grid_cell_start,
		device_grid_cell_end,
		number,
		number_grid_cells);

	//note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
	unmapGLBufferObject(cuda_posvbo_resource);
}

#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif
extern void setCamera(MyShader omyShader);
void ParticleSystem::draw(MyShader& omyShader) {
	omyShader.use();
	setCamera(omyShader);
	glm::mat4 model(1.0f);
	omyShader.setFloat("pointScale",
		SCR_HEIGHT / tanf(45.f * 0.5f * M_PI / 180.0f));
	omyShader.setFloat("pointRadius", params.particleRadius);
	omyShader.setMat4("model", model);

	glBindVertexArray(posVAO);
	glDrawArrays(GL_POINTS, 0, number);
	glBindVertexArray(0);
}

//debugging
void ParticleSystem::dumpParticles(unsigned int start, unsigned int count){
	// debug
	copyArrayFromDevice(host_Position, 0, &cuda_posvbo_resource, sizeof(float) * 4 * count);
	copyArrayFromDevice(host_Velocity, device_Velocity, 0, sizeof(float) * 4 * count);

	for (uint i = start; i<start + count; i++){
		printf("%d: ", i);
		printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", host_Position[i * 4 + 0], host_Position[i * 4 + 1], host_Position[i * 4 + 2], host_Position[i * 4 + 3]);
		printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", host_Velocity[i * 4 + 0], host_Velocity[i * 4 + 1], host_Velocity[i * 4 + 2], host_Velocity[i * 4 + 3]);
	}
}