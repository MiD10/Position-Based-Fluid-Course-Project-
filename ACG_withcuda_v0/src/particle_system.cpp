#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif

#include "../include/helper_math.h"

#include "../include/GL/glew.h"
#include "../include/GLFW/glfw3.h"

#include "../include/particle_system.h"
#include "../include/particle_system.cuh"
#include "../include/particle_kernel.cuh"

using namespace std;

#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif

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
ParticleSystem::ParticleSystem(float dT, unsigned int number_of_particles, int3 gridSize){
	//param setting
	number = number_of_particles;
	params.deltaTime = dT;
	params.numBodies = number;
	params.kernelRadius = 0.1f;
	params.gridSize = gridSize;
	params.worldbound = make_float3(gridSize) * params.kernelRadius * 2;

	//grids&cells
	number_grid_cells = gridSize.x * gridSize.y * gridSize.z;
	params.numCells = number_grid_cells;
	params.cellSize = make_float3(params.kernelRadius * 2.0f);


	//params.boundaryDamping = 0.f;	//new_velocity = velocity * boundaryDamping when bouncing to the wall|floor
	params.gravity = make_float3(0.0f, -9.8f, 0.0f);
	//params.globalDamping = 1.0f; //everytime update(), new_velocity = velocity * globalDamping
	
	//collision
	params.particleRadius = 1.f / 64.f; //particle radius
	params.maxNeighborsPerParticle = 50;
	params.maxParticlesPerCell = 50;


	//pbf
	params.restDensity = 6378.0f; //restDensity
	params.poly6 = 315.f / (64.f * M_PI * pow(params.kernelRadius, 9));
	params.spiky = 45.f / (M_PI *  pow(params.kernelRadius, 6));
	params.numIterations = 4;
	params.relaxation = 600.f;

	//collision
	//params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
	//params.colliderRadius = 0.2f;


	
	//(memory) initialize
	initialize();
}

void ParticleSystem::initialize() {

	//CPU memory allocation : debug only
	host_force = new float[number * 4];
	host_Position = new float[number * 4];
	host_Velocity = new float[number * 4];
	host_density = new float[number];
	host_lamda = new float[number];
	host_delta_Position = new float[number * 4];
	host_neighborsCount = new unsigned int[number];
	memset(host_delta_Position, 0, number * 4 * sizeof(float));
	memset(host_lamda, 0, number * sizeof(float));
	memset(host_density, 0, number * sizeof(float));
	memset(host_force, 0, number * 4 * sizeof(float));
	memset(host_Position, 0, number * 4 * sizeof(float));
	memset(host_Velocity, 0, number * 4 * sizeof(float));
	memset(host_neighborsCount, 0, number * sizeof(unsigned int));

	//grids
	allocateArray((void **)&device_neighbors, number * params.maxNeighborsPerParticle * sizeof(unsigned int));
	allocateArray((void **)&device_neighbors_count, number * sizeof(unsigned int));
	allocateArray((void **)&device_cells, number_grid_cells * params.maxParticlesPerCell * sizeof(unsigned int));
	allocateArray((void **)&device_cells_count, number_grid_cells * sizeof(unsigned int));
	

	unsigned int memSize = sizeof(float) * 4 * number;

	//pbf
	allocateArray((void **)&device_density, number * sizeof(float));
	allocateArray((void **)&device_lamda, number * sizeof(float));
	allocateArray((void **)&device_new_Position, memSize);
	allocateArray((void **)&device_delta_Position, memSize);
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
	//std::cout << "Particle resetting..." << std::endl;
	int p = 0, v = 0;
	for (int i = 0; i < number; i++) {
		host_Position[p++] = params.gridSize.x * 2 * (rand() / (float)RAND_MAX - 0.5);
		host_Position[p++] = params.gridSize.y * 2 * (rand() / (float)RAND_MAX - 0.5);
		host_Position[p++] = params.gridSize.z * 2 * (rand() / (float)RAND_MAX - 0.5);
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
	//std::cout << "Particle reset done!" << std::endl;

	return;
}

void ParticleSystem::resetGrid() {
	//std::cout << "Particle resetting..." << std::endl;
	srand(1973);

	unsigned int i = 0;
	unsigned int size = (int)ceilf(powf((float)params.numBodies, 1.0f / 3.0f));
	//unsigned int size = params.gridSize;
	float spacing = params.kernelRadius * 2.0f;
	float jitter = params.kernelRadius * 0.01f;
	for (unsigned int z = 0; z < size; z++) {
		for (unsigned int y = 0; y < size; y++) {
			for (unsigned int x = 0; x < size; x++, i++) {
				if (i < params.numBodies) {
					//printf("%d, %d, %d, %d\n", i, x, y, z);
					host_Position[i * 4] = (spacing * x) + params.kernelRadius + 2 * (rand() / (float)RAND_MAX - 0.5) * jitter;
					host_Position[i * 4 + 1] = (spacing * y) + params.kernelRadius + 2 * (rand() / (float)RAND_MAX - 0.5) * jitter;
					host_Position[i * 4 + 2] = (spacing * z) + params.kernelRadius + 2 * (rand() / (float)RAND_MAX - 0.5) * jitter;
					host_Position[i * 4 + 3] = 1.0f;

					host_Velocity[i * 4] = 0.0f;
					host_Velocity[i * 4 + 1] = 0.0f;
					host_Velocity[i * 4 + 2] = 0.0f;
					host_Velocity[i * 4 + 3] = 0.0f;
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
	//std::cout << "Particle reset done!" << std::endl;
	return;
}

int cccc = 0;
void ParticleSystem::update(void) {

	device_Position = (float*)mapGLBufferObject(&cuda_posvbo_resource);
	
	setParameters(&params);

	update_fluid(
		device_Velocity,
		device_Position,
		device_new_Position,
		device_density,
		device_lamda,
		device_delta_Position,
		device_neighbors,
		device_neighbors_count,
		device_cells,
		device_cells_count,
		params.numBodies,
		params.numIterations
	);
	
	//note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
	unmapGLBufferObject(cuda_posvbo_resource);
}

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

void ParticleSystem::dumpDensity_Lamda(){
	// dump grid information
	copyArrayFromDevice(host_density, device_density, 0, sizeof(float)*number);
	copyArrayFromDevice(host_lamda, device_lamda, 0, sizeof(float)*number);
	for (uint i = 0; i<number; i++){
		printf("Density = %f | Lamda = %f\n", host_density[i], host_lamda[i]);
	}

	return;
}

void ParticleSystem::dumpLamda() {
	// dump grid information
	copyArrayFromDevice(host_lamda, device_lamda, 0, sizeof(float)*number);

	for (uint i = 0; i<number; i++) {
		printf("%f\n", host_lamda[i]);
	}

	return;
}

void ParticleSystem::dumpDeltaPosition() {
	// dump grid information
	copyArrayFromDevice(host_delta_Position, device_delta_Position, 0, sizeof(float)*number*4);

	for (uint i = 0; i<number; i+=4) {
		printf("%f, %f, %f, %f\n", host_delta_Position[i], host_delta_Position[i+1], host_delta_Position[i+2], host_delta_Position[i+3]);
	}

	return;
}

void ParticleSystem::dumpNeighbors() {
	// dump grid information
	copyArrayFromDevice(host_neighborsCount, device_neighbors_count, 0, sizeof(unsigned int)*number);

	for (uint i = 0; i<number; i += 4) {
		printf("index %d has %d neighbors\n", i, host_neighborsCount[i]);
	}

	return;
}