#pragma once
#include "glad/glad.h"

namespace ConwaysCUDA {

	// We need two VBOs:
	// [0] for alive/dead status
	// [1] for cell age
	enum WORLD_VBOS {
		CELL_STATUS, CELL_AGE, _VBO_COUNT
	};

	bool init(int rows, int cols, GLuint vbos[_VBO_COUNT]);

	void exit();

	void tick();
}
