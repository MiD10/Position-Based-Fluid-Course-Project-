#include "../include/mycamera.h"

MyCamera::MyCamera():cameraPos(glm::vec3(0.0f,0.0f, 5.0f)),
cameraFront(glm::vec3(0.0f, 0.0f, -1.0f)),
cameraUp(glm::vec3(0.0f, 1.0f, 0.0f)),
lastX(SCR_WIDTH / 2), lastY(SCR_HEIGHT / 2),
yaw(-90), pitch(0),
MouseSensitivity(0.05), CameraSensitivity(0.5),
firstMouse(true){
}

void MyCamera::KeyBoardMovement(int Direction){
	float cameraSpeed = 0.05f; // adjust accordingly
	switch (Direction){
	case FORWARD: cameraPos += cameraSpeed * cameraFront; break;
	case BACKWARD: cameraPos -= cameraSpeed * cameraFront; break;
	case LEFTWARD: cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed; break;
	case RIGHTWARD: cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed; break;
	default:;
	}
}

void MyCamera::MouseMovement(double xpos, double ypos){
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}
	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // ע���������෴�ģ���Ϊy�����Ǵӵײ����������������
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