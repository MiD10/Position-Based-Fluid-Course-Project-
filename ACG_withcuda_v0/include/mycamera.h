#ifndef MYCAMERA_H
#define MYCAMERA_H

#include "../include/glm/vec3.hpp" // glm::vec3
#include "../include/glm/vec4.hpp" // glm::vec4
#include "../include/glm/mat4x4.hpp" // glm::mat4
#include "../include/glm/gtc/matrix_transform.hpp" // glm::translate, glm::rotate, glm::scale, glm::perspective

#include "../include/GL/glew.h"
#include "../include/GLFW/glfw3.h"

#define FORWARD 1000
#define BACKWARD 1001
#define LEFTWARD 1002
#define RIGHTWARD 1003

extern unsigned int SCR_WIDTH;
extern unsigned int SCR_HEIGHT;

class MyCamera {
public:
	glm::vec3 cameraPos;
	glm::vec3 cameraFront;
	glm::vec3 cameraUp;
	float lastX, lastY;
	float yaw, pitch;
	float MouseSensitivity;
	float CameraSensitivity;
	bool firstMouse = true;
	MyCamera();
	void KeyBoardMovement(int Direction);
	void MouseMovement(double xpos, double ypos);
};

#endif