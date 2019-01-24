#include "Renderer.h"
#include "ConwaysCUDA.h"

Renderer::Renderer()
{
}

Renderer::~Renderer()
{
	glDeleteBuffers(1, &vbo_quad);
	glDeleteBuffers(1, &vbo_states);
	glDeleteVertexArrays(1, &vao);
}

void Renderer::init(int rows, int cols, const GLbyte * world_state)
{
	this->rows = rows;
	this->cols = cols;

	// General layout:
	// Location 0:
	//   Coordinates of the 4 vertices to make a quad (vec2). 
	//   The actual location of the quad is determined in the vertex shader.
	// Location 1:
	//   The world state given in a 1D array. Each tile has an associated state (alive or dead).
	// The drawing is instanced, so the position and state of the current tile
	// are calculated in the vertex shader based on the index of the current instance.

	float quad[] = {
		0.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 0.0f,
		1.0f, 1.0f
	};

	glGenBuffers(1, &vbo_quad);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_quad);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	
	// Set the VAO's 0-th vertex attribute, pointing it to the currently bound VBO (vbo_quad).
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (void*)(0));

	glGenBuffers(1, &vbo_states);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_states);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLbyte) * rows * cols, world_state, GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(1);
	glVertexAttribIPointer(1, 1, GL_BYTE, sizeof(GLbyte), (void*)(0));
	// Notice the I in ...Attrib'I'Pointer
	// https://www.khronos.org/opengl/wiki/Vertex_Specification
	// The general type of attribute used in the vertex shader must match the 
	// general type provided by the attribute array. This is governed by 
	// which glVertexAttribPointer function you use.

	// Each instance increment the attribute index at 1 by 1.
	glVertexAttribDivisor(1, 1);

	shader = Shader("Shaders/cgol.vert", "Shaders/cgol.frag");
	shader.use();
	shader.setInt("rows", rows);
	shader.setInt("cols", cols);
	shader.setFloat("zoom", 1.0);

	ConwaysCUDA::init(rows, cols, vbo_states);
}

void Renderer::render()
{
	shader.use();
	glBindVertexArray(vao);
	glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, rows * cols);
	glBindVertexArray(0);
}

void Renderer::on_resize(int w, int h)
{
	shader.use();
	shader.setVec2("screen_size", w, h);
}

void Renderer::set_zoom(float zoom)
{
	shader.use();
	shader.setFloat("zoom", zoom);
}

void Renderer::set_camera_center(float x, float y)
{
	shader.use();
	shader.setVec2("camera_center", x, y);
}

void Renderer::set_world_grid(const GLbyte * world)
{
	glBindBuffer(GL_ARRAY_BUFFER, vbo_states);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLbyte) * rows * cols, world, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
