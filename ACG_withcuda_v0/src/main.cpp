#ifdef _WIN32
#ifndef _WINDOWS_HAS_INCLUDED
#define _WINDOWS_HAS_INCLUDED
#include <windows.h>
#endif
#endif

#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif

#include "../include/GL/glew.h"
#include "../include/GLFW/glfw3.h"

#include "../include/myinclude.h"

#include "../include/myshader.h"
#include "../include/mycamera.h"
#include "../include/mymesh.h"
#include "../include/mymodel.h"

#include "../include/particle_system.h"

using namespace std;

//opengl
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
int draw_opengl(void);

//define a universal camera
MyCamera camera;
void setCamera(MyShader omyShader) {
	//glm::mat4 model = glm::mat4(1.0f);
	//model = glm::scale(glm::mat4(1.0f), glm::vec3(0.005f));
	//model = glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	glm::mat4 view = glm::mat4(1.0f);
	view = glm::lookAt(camera.cameraPos, camera.cameraPos + camera.cameraFront, camera.cameraUp);
	glm::mat4 projection = glm::mat4(1.0f);
	projection = glm::perspective<float>(glm::radians(45.0f), SCR_WIDTH / SCR_HEIGHT, 0.1f, 200.0f);
	//omyShader.setMat4("model", model);
	omyShader.setMat4("view", view);
	omyShader.setMat4("projection", projection);
}

void setParams(SimParams& params, int number) {
	//param setting
	params.deltaTime = 0.5f;
	params.numBodies = number;
	params.particleRadius = 1.f / 64.f; //particle radius
	params.gridSize = make_int3(64,64,64);
	params.number_grid_cells = params.gridSize.x * params.gridSize.y * params.gridSize.z;
	params.cellSize = make_float3(params.particleRadius * 2.0f, params.particleRadius * 2.0f, params.particleRadius * 2.0f);
	params.worldBounds = make_float3(
		params.gridSize.x * params.particleRadius,
		params.gridSize.y * params.particleRadius,
		params.gridSize.z * params.particleRadius);

	params.boundaryDamping = -0.f;	//new_velocity = velocity * boundaryDamping when bouncing to the wall|floor
	params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
	params.globalDamping = 1.0f; //everytime update(), new_velocity = velocity * globalDamping

	//collision
	//params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
	//params.colliderRadius = 0.2f;
	params.spring = 0.5f;
	params.damping = 0.02f;
	params.shear = 0.1f;
	params.attraction = 0.0f;

	
}


// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
bool global_reset = false;
void processInput(GLFWwindow *window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	float cameraSpeed = 0.05f; // adjust accordingly
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.KeyBoardMovement(FORWARD);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.KeyBoardMovement(BACKWARD);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.KeyBoardMovement(LEFTWARD);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.KeyBoardMovement(RIGHTWARD);
	if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
		global_reset = true;
	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
		camera.MouseSensitivity += (float)0.002;
	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
		camera.MouseSensitivity -= (float)0.002;
}
//process for mouse movement
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
	camera.MouseMovement(xpos, ypos);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

int main(void) {

	int num;
	std::cin >> num;

	// glfw: initialize and configure
	// ------------------------------
	cout << "INITIALIZING..." << endl;
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// start GLEW extension handler
	glewExperimental = GL_TRUE;
	glewInit();

	// get version info
	const GLubyte* renderer = glGetString(GL_RENDERER); // get renderer string
	const GLubyte* version = glGetString(GL_VERSION); // version as a string
	printf("Renderer: %s\n", renderer);
	printf("OpenGL version supported %s\n", version);

	cout << "INITIALIZATION FINISHED" << endl;

	//ENABLE DEPTH
	//Point Sprite
	glEnable(GL_DEPTH_TEST); // enable depth-testing
	glDepthFunc(GL_LESS); // depth-testing interprets a smaller value as "closer"
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_SMOOTH);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	glClearColor(0, 0, 0, 1);

	//shader preparation
	MyShader omyShader("Shader\\SphereVertexShader.gl", "Shader\\ShpereFragShader.gl");

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(window, mouse_callback);

	SimParams temp;
	setParams(temp, num);
	ParticleSystem tester(temp);

	tester.resetGrid();
	std::cout << "here" << std::endl;
	while (!glfwWindowShouldClose(window))
	{
		if (global_reset) {
			tester.resetGrid();
			global_reset = false;
		}
		// input
		// -----
		processInput(window);

		// render
		// ------
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		tester.draw(omyShader);
		tester.update();
		//draw
		/*omyShader.use();
		setCamera(omyShader);
		glm::mat4 model(1.0f);
		omyShader.setFloat("pointScale", SCR_HEIGHT / tanf(45.f*0.5f*(float)3.14159 / 180.0f));
		omyShader.setFloat("pointRadius", 0.5);
		omyShader.setMat4("model", model);*/
		
		/*for (int i = 0; i < omyMS.objects.size(); i++) {
			glm::mat4 model(1.0f);
			model = glm::translate(model, omyMS.objects[i].position);
			model = glm::scale(model, glm::vec3(omyMS.objects[i].scale));
			omyShader.setMat4("model", model);
			omyModel.Draw(omyShader);
		}*/


		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	//system("pause");
	return 0;
}