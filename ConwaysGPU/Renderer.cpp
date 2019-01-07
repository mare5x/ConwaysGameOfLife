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

void Renderer::init(int rows, int cols)
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

	float* states = new float[rows * cols];
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			if ((row + col) % 4 == 0)
				states[row * cols + col] = 1.0f;
			else
				states[row * cols + col] = 0;
		}
	}

	glGenBuffers(1, &vbo_states);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_states);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * rows * cols, states, GL_DYNAMIC_DRAW);

	delete[] states;

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT), (void*)(0));

	glVertexAttribDivisor(1, 1);

	shader = Shader("Shaders/cgol.vert", "Shaders/cgol.frag");
	shader.use();
	shader.setInt("rows", rows);
	shader.setInt("cols", cols);

	ConwaysCUDA::init(rows, cols, vbo_states);
}

void Renderer::render()
{
	shader.use();
	glBindVertexArray(vao);
	glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, rows * cols);
	glBindVertexArray(0);
}