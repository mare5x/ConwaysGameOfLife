#pragma once
#include "Shaders\Shader.h"
#include "glad\glad.h"

class Renderer {
public:
	Renderer();
	~Renderer();

	void init(int rows, int cols, const float* world_state);

	void render();

	void on_resize(int w, int h);

	void set_zoom(float zoom);

	void set_world_grid(const float* world);
private:
	GLuint vao, vbo_quad, vbo_states;
	Shader shader;

	int rows, cols;
};