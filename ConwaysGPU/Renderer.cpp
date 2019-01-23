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

void Renderer::init(int rows, int cols, const float* world_state)
{
	this->rows = rows;
	this->cols = cols;

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
	
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (void*)(0));

	glGenBuffers(1, &vbo_states);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_states);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * rows * cols, world_state, GL_DYNAMIC_DRAW);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT), (void*)(0));

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

void Renderer::set_world_grid(const float * world)
{
	glBindBuffer(GL_ARRAY_BUFFER, vbo_states);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * rows * cols, world, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
