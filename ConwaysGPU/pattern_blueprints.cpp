#include "pattern_blueprints.h"
#include <algorithm>

using ConwaysCUDA::WORLD_T;

namespace still_lifes {
	Pattern block = {
		"11"
		"11", 2, 2
	};

	Pattern beehive = {
		"0110"
		"1001"
		"0110", 3, 4
	};

	Pattern loaf = {
		"0110"
		"1001"
		"0101"
		"0010", 4, 4
	};

	Pattern boat = {
		"110"
		"101"
		"010", 3, 3
	};

	Pattern tub = {
		"010"
		"101"
		"010", 3, 3
	};
}

namespace oscillators {
	Pattern blinker = {
		"111", 1, 3
	};

	Pattern toad = {
		"0111"
		"1110", 2, 4
	};

	Pattern beacon = {
		"1100"
		"1100"
		"0011"
		"0011", 4, 4
	};

	Pattern pulsar = {
		"0011100011100"
		"0000000000000"
		"1000010100001"
		"1000010100001"
		"1000010100001"
		"0011100011100"
		"0000000000000"
		"0011100011100"
		"1000010100001"
		"1000010100001"
		"1000010100001"
		"0000000000000"
		"0011100011100", 13, 13
	};

	Pattern pentadecathlon = {
		"111"
		"101"
		"111"
		"111"
		"111"
		"111"
		"101"
		"111", 8, 3
	};
}

namespace spaceships {
	Pattern glider = {
		"010"
		"001"
		"111", 3, 3
	};

	Pattern lwss = {
		"01111"
		"10001"
		"00001"
		"10010", 4, 5
	};

	Pattern mwss = {
		"111110"
		"100001"
		"100000"
		"010001"
		"000100", 5, 6
	};

	Pattern hwss = {
		"1111110"
		"1000001"
		"1000000"
		"0100001"
		"0001100", 5, 7
	};
}

namespace others {
	Pattern r_pentomino = {
		"011"
		"110"
		"010", 3, 3
	};

	Pattern diehard = {
		"00000010"
		"11000000"
		"01000111", 3, 8
	};

	Pattern acorn = {
		"0100000"
		"0001000"
		"1100111", 3, 7
	};


	Pattern mid_gosper = {
		"000000000000001"
		"000000000000101"
		"001100000011000"
		"010001000011000"
		"100000100011000"
		"100010110000101"
		"100000100000001"
		"010001000000000"
		"001100000000000", 9, 15
	};

	MultiPattern gosper_glider_gun(
		{ &still_lifes::block, &mid_gosper, &still_lifes::block },
		{ -4, 0, -2 },
		{ 0, 10, 34 }
	);

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

	Pattern inf_engine_1 = {
		"00000010"
		"00001011"
		"00001010"
		"00001000"
		"00100000"
		"10100000", 6, 8
	};

	Pattern inf_engine_2 = {
		"11101"
		"10000"
		"00011"
		"01101"
		"10101", 5, 5
	};

	Pattern inf_engine_3 = {
		"111111110111110001110000001111111011111", 1, 39
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

	MultiPattern multi_multi_pattern(
		{ &multi_test_pattern, &multi_test_pattern, &multi_test_pattern, &multi_test_pattern },
		{ 0, 0, -multi_test_pattern.height() - 2, -multi_test_pattern.height() - 2 },
		{ 0, multi_test_pattern.width() + 2, 0, multi_test_pattern.width() + 2 }
	);
}


std::vector<std::vector<Blueprint*> > pattern_blueprints::all_patterns = {
	{   // STILL LIFES
		&still_lifes::block,
		&still_lifes::beehive,
		&still_lifes::loaf,
		&still_lifes::boat,
		&still_lifes::tub
	},

	{	// OSCILLATORS
		&oscillators::blinker,
		&oscillators::toad,
		&oscillators::beacon,
		&oscillators::pulsar,
		&oscillators::pentadecathlon
	},
	
	{	// SPACESHIPS
		&spaceships::glider,
		&spaceships::lwss,
		&spaceships::mwss,
		&spaceships::hwss
	},

	{	// OTHERS
		&others::r_pentomino,
		&others::diehard,
		&others::acorn,
		&others::gosper_glider_gun,
		&others::shooter_pattern,
		&others::inf_engine_1,
		&others::inf_engine_2,
		&others::inf_engine_3,
	}
};

std::vector<Blueprint*>& pattern_blueprints::still_lifes = 
	pattern_blueprints::all_patterns[pattern_blueprints::STILL_LIFES];

std::vector<Blueprint*>& pattern_blueprints::oscillators =
	pattern_blueprints::all_patterns[pattern_blueprints::OSCILLATORS];

std::vector<Blueprint*>& pattern_blueprints::spaceships =
	pattern_blueprints::all_patterns[pattern_blueprints::SPACESHIPS];

std::vector<Blueprint*>& pattern_blueprints::others =
	pattern_blueprints::all_patterns[pattern_blueprints::OTHERS];


// Transform the given [row, col] in blueprint coordinates 
// to rotated blueprint coordinates. 
// Blueprint coordinate origin is the top left corner, with rows increasing
// downwards.
__host__ __device__ void get_rotated_coordinates(int rotation, int height, int width, int* row, int* col)
{
	// Work it out on paper to see that this recursive formula works.
	while (rotation > 0) {
		int tmp = *row;
		*row = width - *col - 1;
		*col = tmp;

		tmp = height;
		height = width;
		width = tmp;
		rotation -= 90;
	}
}

void Pattern::build(WORLD_T * world, int w_rows, int w_cols, int row, int col) const
{
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			char val = pattern[i * cols + j] - '0';

			// Apply the rotation if necessary.
			int rot_i = i;
			int rot_j = j;
			if (rotation > 0)
				get_rotated_coordinates(&rot_i, &rot_j);

			// (0,0) is at the bottom left
			int prow = (w_rows + row - rot_i) % w_rows;
			int pcol = (w_cols + col + rot_j) % w_cols;
			int idx = prow * w_cols + pcol;
			world[idx] = val;
		}
	}
}

__host__ __device__ void Pattern::get_rotated_coordinates(int * row, int * col) const
{
	::get_rotated_coordinates(rotation, rows, cols, row, col);
}

void MultiPattern::build(WORLD_T * world, int rows, int cols, int row, int col) const
{
	// We have to take the rotation into account.
	// Rotate the individual components (done in set_rotation) 
	// and also correct the row and col offsets.
	for (int i = 0; i < blueprints.size(); ++i) {
		int row_offset = row_offsets[i];
		int col_offset = col_offsets[i];

		// A more elegant solution for not having to deal with 
		// rotated offsets would be to apply the rotation only  
		// when building Patterns. However that would require
		// each Pattern to know about the dimensions of 
		// it's root parent MultiPattern. Maybe later? // TODO
		get_rotated_offset(blueprints[i], row_offset, col_offset);

		blueprints[i]->build(world, rows, cols, row + row_offset, col + col_offset);
	}
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

void MultiPattern::set_rotation(int deg) 
{
	for (Blueprint* blueprint : blueprints)
		blueprint->set_rotation(deg);
}

void MultiPattern::get_rotated_offset(const Blueprint* blueprint, int & row_offset, int & col_offset) const
{
	int rotation = get_rotation() % 360;

	// -1 because get_rotated_coordinates assumes y down is positive.
	row_offset *= -1;
	::get_rotated_coordinates(rotation, height(), width(), &row_offset, &col_offset);

	// The point was rotated, but now we have to bring it back to the 
	// top left corner of the given blueprint:
	switch (rotation) {
	case 90: 
		row_offset -= blueprint->width() + 1;
		break;
	case 180: 
		row_offset -= blueprint->height() + 1;
		col_offset -= blueprint->width() + 1;
		break;
	case 270: 
		col_offset -= blueprint->height() + 1;
		break;
	}

	row_offset *= -1;
}
