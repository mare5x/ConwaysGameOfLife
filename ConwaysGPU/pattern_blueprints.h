#pragma once
#include <vector>
#include "ConwaysCUDA.h"
#include <cuda_runtime.h>

// Base abstract interface for describing pattern blueprints.
struct Blueprint {
	enum BlueprintType {
		Pattern, MultiPattern
	};

	virtual ~Blueprint() { }
	// Place the pattern so that the top left corner is at [row, col].
	virtual void build(ConwaysCUDA::WORLD_T* world, int rows, int cols, int row, int col) const = 0;
	virtual int width() const = 0;
	virtual int height() const = 0;

	// Rotate the blueprint deg degrees counter clockwise. Must be a multiple of 90.
	virtual void set_rotation(int deg) = 0;
	virtual int get_rotation() const = 0;

	virtual BlueprintType type() const = 0;
};

struct Pattern : Blueprint {
	virtual void build(ConwaysCUDA::WORLD_T* world, int w_rows, int w_cols, int row, int col) const override;
	virtual int width() const override { return cols; }
	virtual int height() const override { return rows; }

	virtual void set_rotation(int deg) override { rotation = deg % 360; }
	virtual int get_rotation() const override { return rotation; }

	__host__ __device__ void get_rotated_coordinates(int* row, int* col) const;

	virtual BlueprintType type() const override { return BlueprintType::Pattern; }

	Pattern(const char* pattern, int rows, int cols)
		: pattern(pattern)
		, rows(rows)
		, cols(cols)
	{ }

	// A string of 1s and 0s, where each row is width chars long.
	const char* pattern;
	int rows, cols;
	int rotation = 0;
};

struct MultiPattern : Blueprint {
	virtual void build(ConwaysCUDA::WORLD_T* world, int rows, int cols, int row, int col) const override;
	virtual int width() const override;
	virtual int height() const override;

	virtual void set_rotation(int deg) override;
	virtual int get_rotation() const override { return blueprints.front()->get_rotation(); }

	void get_rotated_offset(const Blueprint* blueprint, int & row_offset, int & col_offset) const;

	virtual BlueprintType type() const override { return BlueprintType::MultiPattern; }

	MultiPattern(std::initializer_list<Blueprint*> patterns, 
			     std::initializer_list<int> row_offsets, 
			     std::initializer_list<int> col_offsets) 
		: blueprints(patterns)
		, row_offsets(row_offsets)
		, col_offsets(col_offsets) 
	{ }

	std::vector<Blueprint*> blueprints;
	std::vector<int> row_offsets, col_offsets;
};


namespace pattern_blueprints {
	enum PatternCategory {
		STILL_LIFES, OSCILLATORS, SPACESHIPS, OTHERS, _N_PATTERN_CATEGORIES
	};
	extern std::vector<std::vector<Blueprint*> > all_patterns;

	// Patterns based on https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
	extern std::vector<Blueprint*>& still_lifes;
	extern std::vector<Blueprint*>& oscillators;
	extern std::vector<Blueprint*>& spaceships;
	extern std::vector<Blueprint*>& others;
}
