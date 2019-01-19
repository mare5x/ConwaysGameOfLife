#pragma once
#include "Shaders\Shader.h"
#include "glad\glad.h"

class Renderer {
public:
	Renderer();
	~Renderer();

	void init(int rows, int cols);

	void render();

	void on_resize(int w, int h);

	void set_zoom(float zoom);
private:
	GLuint vao, vbo_quad, vbo_states;
	Shader shader;

	int rows, cols;
};