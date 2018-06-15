#ifndef MYSHADER_H
#define MYSHADER_H

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "../include/GL/glew.h"
#include "../include/GLFW/glfw3.h"

#include "../include/glm/vec3.hpp" // glm::vec3
#include "../include/glm/vec4.hpp" // glm::vec4
#include "../include/glm/mat4x4.hpp" // glm::mat4
#include "../include/glm/gtc/matrix_transform.hpp" // glm::translate, glm::rotate, glm::scale, glm::perspective

class MyShader
{
public:
	GLint ID;

	// ��������ȡ��������ɫ��
	MyShader(const GLchar* VertexShaderPath, const GLchar* FragmentShaderPath);
	// ʹ��/�������
	void use();
	// uniform���ߺ���
	void setBool(const std::string &name, bool value) const;
	void setInt(const std::string &name, int value) const;
	void setFloat(const std::string &name, float value) const;
	void setMat4(const std::string &name, glm::mat4 value) const;
};

#endif