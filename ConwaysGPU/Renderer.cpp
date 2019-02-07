#include "Renderer.h"

using ConwaysCUDA::CELL_AGE_T;
using ConwaysCUDA::CELL_STATUS_T;


Renderer::~Renderer()
{
	glDeleteBuffers(1, &vbo_quad);
	glDeleteBuffers(ConwaysCUDA::_VBO_COUNT, vbo_states);
	glDeleteVertexArrays(1, &vao);
}

void Renderer::init(int rows, int cols)
{
	this->rows = rows;
	this->cols = cols;

	// General layout:
	// Location 0:
	//   Coordinates of the 4 vertices to make a quad (vec2). 
	//   The actual location of the quad is determined in the vertex shader.
	// Location 1:
	//   The world state given in a 1D array. Each tile has an associated state (alive or dead).
	// Location 2:
	//   The age of each cell in a 1D array.
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

	glGenBuffers(ConwaysCUDA::_VBO_COUNT, vbo_states);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_states[ConwaysCUDA::CELL_STATUS]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(CELL_STATUS_T) * rows * cols, NULL, GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(1);
	glVertexAttribIPointer(1, 1, GL_BYTE, sizeof(CELL_STATUS_T), (void*)(0));
	// Notice the I in ...Attrib'I'Pointer
	// https://www.khronos.org/opengl/wiki/Vertex_Specification
	// The general type of attribute used in the vertex shader must match the 
	// general type provided by the attribute array. This is governed by 
	// which glVertexAttribPointer function you use.

	glBindBuffer(GL_ARRAY_BUFFER, vbo_states[ConwaysCUDA::CELL_AGE]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(CELL_AGE_T) * rows * cols, NULL, GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(2);
	glVertexAttribIPointer(2, 1, GL_INT, sizeof(CELL_AGE_T), (void*)(0));

	// Each instance increment the attribute index at 1 and 2 by 1.
	glVertexAttribDivisor(1, 1);
	glVertexAttribDivisor(2, 1);

	shader = Shader("Shaders/cgol.vert", "Shaders/cgol.frag");
	shader.use();
	shader.setInt("rows", rows);
	shader.setInt("cols", cols);
	shader.setFloat("zoom", 1.0);
	shader.setBool("show_grid", true);

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

void Renderer::set_world_grid(const CELL_STATUS_T * world)
{
	glBindBuffer(GL_ARRAY_BUFFER, vbo_states[ConwaysCUDA::CELL_STATUS]);
	// When updating an existing buffer use glBufferSubData instead of glBufferData!
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(CELL_STATUS_T) * rows * cols, world);

	// CUDA code works mostly with the CELL_AGE vbo, so we have to populate it ...
	glBindBuffer(GL_ARRAY_BUFFER, vbo_states[ConwaysCUDA::CELL_AGE]);
	CELL_AGE_T* cell_ages = (CELL_AGE_T*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	for (int i = 0; i < rows * cols; ++i)
		cell_ages[i] = world[i];
	glUnmapBuffer(GL_ARRAY_BUFFER);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Renderer::set_grid_visibility(bool visible)
{
	shader.use();
	shader.setBool("show_grid", visible);
}

void Renderer::toggle_cell(int row, int col)
{
	glBindBuffer(GL_ARRAY_BUFFER, vbo_states[ConwaysCUDA::CELL_AGE]);
	CELL_AGE_T* cell_ages = (CELL_AGE_T*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_states[ConwaysCUDA::CELL_STATUS]);
	CELL_STATUS_T* cell_status = (CELL_STATUS_T*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);

	CELL_STATUS_T val = cell_status[row * cols + col];
	cell_status[row * cols + col] = (val > 0 ? 0 : 1);
	cell_ages[row * cols + col] = (val > 0 ? 0 : 1);

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_states[ConwaysCUDA::CELL_AGE]);
	glUnmapBuffer(GL_ARRAY_BUFFER);
}
