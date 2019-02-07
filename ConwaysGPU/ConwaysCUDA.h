#pragma once
#include "glad/glad.h"

namespace ConwaysCUDA {

	// We need two VBOs:
	// [0] for alive/dead status
	// [1] for cell age
	enum WORLD_VBOS {
		CELL_STATUS, CELL_AGE, _VBO_COUNT
	};

	// Type define the OpenGL types we'll be using to allow easier modification.
	typedef GLbyte CELL_STATUS_T;
	typedef GLint CELL_AGE_T;
	typedef CELL_STATUS_T WORLD_T;  // Make the intent clearer ...

	bool init(int rows, int cols, GLuint vbos[_VBO_COUNT]);

	void exit();

	void tick();

	//
	//void build_pattern(const Pattern& blueprint);

	// Toggle the state of the given cell in "real-time".
	void toggle_cell(int row, int col);
}
