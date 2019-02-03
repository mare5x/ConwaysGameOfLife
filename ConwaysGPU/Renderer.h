#pragma once
#include "Shaders\Shader.h"
#include "glad\glad.h"
#include "ConwaysCUDA.h"

class Renderer {
public:
	Renderer() = default;
	~Renderer();

	void init(int rows, int cols, const GLbyte * world_state);

	void render();

	void on_resize(int w, int h);

	void set_zoom(float zoom);

	// x and y must be in range [0, 1].
	void set_camera_center(float x, float y);

	void set_world_grid(const GLbyte * world);
	void set_grid_visibility(bool visible);
private:
	GLuint vao, vbo_quad;
	GLuint vbo_states[ConwaysCUDA::_VBO_COUNT];
	Shader shader;

	int rows, cols;
};