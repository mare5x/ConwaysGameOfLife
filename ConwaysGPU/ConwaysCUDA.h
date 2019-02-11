#pragma once
#include "glad/glad.h"

// Forward declarations.
struct Blueprint;
struct Pattern;
struct MultiPattern;

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

	// For performance reasons, wrap all functions below with these two functions.
	void start_interop();
	void stop_interop();

	void tick();

	// Build the given blueprint pattern in "real-time", where the
	// top left corner of the pattern is at [row, col].
	void set_pattern(const Blueprint& blueprint, int row, int col);

	// Set or unset the given blueprint pattern in hover mode.
	void set_hover_pattern(const Blueprint& blueprint, int row, int col, bool hover);

	// Toggle the state of the given cell in "real-time".
	void toggle_cell(int row, int col);
}
