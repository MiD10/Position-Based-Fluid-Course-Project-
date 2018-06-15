#include "../include/mymesh.h"

using namespace std;

MyMesh::MyMesh(vector<Vertex> v, vector<unsigned int> i, vector<Texture> t){
	vertices = v;
	indices = i;
	textures = t;
	setupMesh();
}

void MyMesh::setupMesh(){
	cout << "BINDING VBO & VAO..." << endl;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

	//position
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	//color
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)offsetof(Vertex, Normal));
	glEnableVertexAttribArray(1);
	//TexCoord
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)offsetof(Vertex, TexCoords));
	glEnableVertexAttribArray(2);
	cout << "VAO & VBO BINDING FINISHED" << endl;
}

void MyMesh::Draw(MyShader omyShader){
	for (int i = 0; i < textures.size(); i++){
		glActiveTexture(GL_TEXTURE0 + i); // 在绑定之前激活相应的纹理单元
		//omyShader.setInt(textures[i].name, i);
		glBindTexture(GL_TEXTURE_2D, textures[i].id);
	}
	glActiveTexture(GL_TEXTURE0); // 在绑定之前激活相应的纹理单元
	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}