#pragma once
#include "glad/glad.h"

namespace ConwaysCUDA {
	bool init(int rows, int cols, GLuint vbo);

	void exit();

	void tick();
}
