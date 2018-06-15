#ifndef MYMESH_H
#define MYMESH_H

#include <vector>

#include "../include/myshader.h"

//assimp to load model
#include "../include/assimp/Importer.hpp"
#include "../include/assimp/scene.h"
#include "../include/assimp/postprocess.h"

using namespace std;

struct Vertex{
	glm::vec3 Position;
	glm::vec3 Normal;
	glm::vec2 TexCoords;
};

struct Texture{
	unsigned int id;
	string type;
	aiString path;
};

class MyMesh{
public:
	vector<Vertex> vertices;
	vector<unsigned int> indices;
	vector<Texture> textures;
	MyMesh(vector<Vertex> vertices, vector<unsigned int> indices, vector<Texture> textures);
	void Draw(MyShader shader);
private:
	unsigned int VAO, VBO, EBO;
	void setupMesh();
};

#endif