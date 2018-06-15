#include "../include/myshader.h"

using namespace std;

MyShader::MyShader(const GLchar* VertexShaderPath, const GLchar* FragmentShaderPath){
	cout << "BEGIN READING CODE IN..." << endl;
	ifstream VSfile;
	ifstream FSfile;
	VSfile.open(VertexShaderPath);
	if (!VSfile.is_open()){
		cout << "ERROR OPENING VERTEX_SHADER_FILE" << endl;
		system("pause");
		exit(0);
	}
	FSfile.open(FragmentShaderPath);
	if (!VSfile.is_open()){
		cout << "ERROR OPENING FRAGMENT_SHADER_FILE" << endl;
		system("pause");
		exit(0);
	}
	
	stringstream VSstream;
	stringstream FSstream;
	string VSstring;
	string FSstring;
	VSstream << VSfile.rdbuf();
	FSstream << FSfile.rdbuf();
	VSstring = VSstream.str();
	FSstring = FSstream.str();
	VSfile.close();
	FSfile.close();

	const char* VScode;
	const char* FScode;
	VScode = VSstring.c_str();
	FScode = FSstring.c_str();
	cout << "CODE IS SUCCESSFULLY READED IN" << endl;
	
	
	//COMPILATION
	cout << "BEGIN COMPILING SHADERS..." << endl; 
	unsigned int vertex, fragment;
	int success;
	char infoLog[512];

	//VERTEX SHADER
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &VScode, NULL);
	glCompileShader(vertex);
	//ERROR CHECKING
	glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
	if (!success){
		glGetShaderInfoLog(vertex, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	};

	//FRAGMENT SHADER
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &FScode, NULL);
	glCompileShader(fragment);
	//ERROR CHECKING
	glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
	if (!success){
		glGetShaderInfoLog(fragment, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	};
	cout << "SHADER COMPILATION FINISHED" << endl;


	//ATTACHING
	cout << "BEGIN ATTACHING SHADERS..." << endl;
	ID = glCreateProgram();
	glAttachShader(ID, vertex);
	glAttachShader(ID, fragment);
	glLinkProgram(ID);
	//ERROR CHECKING
	glGetProgramiv(ID, GL_LINK_STATUS, &success);
	if (!success){
		glGetProgramInfoLog(ID, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}
	glDeleteShader(vertex);
	glDeleteShader(fragment);
	cout << "SHADER ATTACHING FINISHED" << endl;

	cout << "SHADER PROGRAM ID: " << ID << endl;
}

void MyShader::use(){
	glUseProgram(ID);
}

void MyShader::setBool(const std::string &name, bool value) const{
	glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
}
void MyShader::setInt(const std::string &name, int value) const{
	glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}
void MyShader::setFloat(const std::string &name, float value) const{
	glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}
void MyShader::setMat4(const std::string &name, glm::mat4 value) const{
	glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &value[0][0]);
}

