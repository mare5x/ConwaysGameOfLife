#include "pattern_blueprints.h"
#include <algorithm>

using ConwaysCUDA::WORLD_T;

namespace {
	Pattern blueprints[] = {
		{
			"0000000010"
			"0000000101"
			"0000001011"
			"1100011011"
			"1100001011"
			"0000000101"
			"0000000010", 7, 10
		},
		{
			"0011"
			"0101"
			"1000"
			"1001"
			"1000"
			"0101"
			"0011", 7, 4 
		},
		{
			"0110000"
			"1001011"
			"0110011", 3, 7
		}
	};

	// Remember:
	// The coordinate system's origin is in the lower left corner.
	// So to move a pattern down, the offset must be negative!

	MultiPattern shooter_pattern(
		{ &blueprints[0], &blueprints[1], &blueprints[2] },
		{ 0, -2, -4 },
		{ 0, 23, 29 }
	);

	MultiPattern multi_test_pattern(
		{ &shooter_pattern, &shooter_pattern, &shooter_pattern, &shooter_pattern },
		{ 0, 0, -shooter_pattern.height() - 2, -shooter_pattern.height() - 2 },
		{ 0, shooter_pattern.width() + 2, 0, shooter_pattern.width() + 2 }
	);
}

const Blueprint& pattern_blueprints::shooter_pattern = shooter_pattern;

std::vector<const Blueprint*> pattern_blueprints::all_patterns{
	&pattern_blueprints::shooter_pattern,
	&blueprints[0],
	&blueprints[1],
	&blueprints[2],
	&multi_test_pattern
};

void Pattern::build(WORLD_T * world, int w_rows, int w_cols, int row, int col) const
{
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			char val = pattern[i * cols + j] - '0';
			// (0,0) is at the bottom left
			int prow = (w_rows + row - i) % w_rows;
			int pcol = (w_cols + col + j) % w_cols;
			int idx = prow * w_cols + pcol;
			world[idx] = val;
		}
	}
}

void MultiPattern::build(WORLD_T * world, int rows, int cols, int row, int col) const
{
	for (int i = 0; i < blueprints.size(); ++i)
		blueprints[i]->build(world, rows, cols, row + row_offsets[i], col + col_offsets[i]);
}

int MultiPattern::width() const
{
	int left = 0;
	int right = 0;
	for (int i = 0; i < blueprints.size(); ++i) {
		left = std::min(left, col_offsets[i]);
		right = std::max(right, col_offsets[i] + blueprints[i]->width());
	}
	return right - left;
}

int MultiPattern::height() const
{
	int top = 0;
	int bot = 0;
	for (int i = 0; i < blueprints.size(); ++i) {
		top = std::max(top, row_offsets[i]);
		bot = std::min(bot, row_offsets[i] - blueprints[i]->height());
	}
	return top - bot;
}
