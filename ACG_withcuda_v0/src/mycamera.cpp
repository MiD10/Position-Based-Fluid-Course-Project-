#include "../include/mycamera.h"

MyCamera::MyCamera() :cameraPos(glm::vec3(-8.505245f, 4.543453f, 6.628063f)),
cameraFront(glm::vec3(0.935821f, -0.253758f, -0.244635f)),
cameraUp(glm::vec3(0.0f, 1.0f, 0.0f)),
lastX(SCR_WIDTH / 2), lastY(SCR_HEIGHT / 2),
yaw(-14.649985), pitch(-14.699993),
MouseSensitivity(0.05), CameraSensitivity(5),
firstMouse(true) {
}

void MyCamera::KeyBoardMovement(int Direction) {
	float cameraSpeed = 0.3f; // adjust accordingly
	switch (Direction) {
	case FORWARD: cameraPos += cameraSpeed * cameraFront; break;
	case BACKWARD: cameraPos -= cameraSpeed * cameraFront; break;
	case LEFTWARD: cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed; break;
	case RIGHTWARD: cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed; break;
	default:;
	}
}

void MyCamera::MouseMovement(double xpos, double ypos) {
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}
	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // 注意这里是相反的，因为y坐标是从底部往顶部依次增大的
	lastX = xpos;
	lastY = ypos;
	xoffset *= MouseSensitivity;
	yoffset *= MouseSensitivity;
	yaw += xoffset;
	pitch += yoffset;
	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;

	glm::vec3 front = glm::vec3(1.0f);
	front.x = cos(glm::radians(pitch)) * cos(glm::radians(yaw));
	front.y = sin(glm::radians(pitch));
	front.z = cos(glm::radians(pitch)) * sin(glm::radians(yaw));
	cameraFront = glm::normalize(front);
}
