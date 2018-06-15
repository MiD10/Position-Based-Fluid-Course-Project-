#ifndef MYMODEL_H
#define MYMODEL_H

#include <map>

#include "../include/mymesh.h"

#include "../include/stb_image.h"

using namespace std;

class MyModel{
public:
	/*  ����   */
	MyModel(string path);
	void Draw(MyShader shader);
private:
	/*  ģ������  */
	vector<MyMesh> meshes;
	string directory;
	/*  ����   */
	void loadModel(string path);
	void processNode(aiNode *node, const aiScene *scene);
	MyMesh processMesh(aiMesh *mesh, const aiScene *scene);
	vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type, string typeName);
	int TextureFromFile(const char* path, const string& directory);
};

#endif