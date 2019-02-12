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


std::vector<std::vector<const Blueprint*> > pattern_blueprints::all_patterns = {
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

std::vector<const Blueprint*>& pattern_blueprints::still_lifes = 
	pattern_blueprints::all_patterns[pattern_blueprints::STILL_LIFES];

std::vector<const Blueprint*>& pattern_blueprints::oscillators =
	pattern_blueprints::all_patterns[pattern_blueprints::OSCILLATORS];

std::vector<const Blueprint*>& pattern_blueprints::spaceships =
	pattern_blueprints::all_patterns[pattern_blueprints::SPACESHIPS];

std::vector<const Blueprint*>& pattern_blueprints::others =
	pattern_blueprints::all_patterns[pattern_blueprints::OTHERS];

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
