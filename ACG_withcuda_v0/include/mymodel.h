#ifndef MYMODEL_H
#define MYMODEL_H

#include <map>

#include "../include/mymesh.h"

#include "../include/stb_image.h"

using namespace std;

class MyModel{
public:
	/*  函数   */
	MyModel(string path);
	void Draw(MyShader shader);
private:
	/*  模型数据  */
	vector<MyMesh> meshes;
	string directory;
	/*  函数   */
	void loadModel(string path);
	void processNode(aiNode *node, const aiScene *scene);
	MyMesh processMesh(aiMesh *mesh, const aiScene *scene);
	vector<Texture> loadMaterialTextures(aiMaterial *mat, aiTextureType type, string typeName);
	int TextureFromFile(const char* path, const string& directory);
};

#endif