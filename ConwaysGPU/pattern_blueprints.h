#pragma once
#include <vector>
#include "ConwaysCUDA.h"

// Base abstract interface for describing pattern blueprints.
struct Blueprint {
	virtual ~Blueprint() { }
	// Place the pattern so that the top left corner is at [row, col].
	virtual void build(ConwaysCUDA::WORLD_T* world, int rows, int cols, int row, int col) const = 0;
	virtual int width() const = 0;
	virtual int height() const = 0;
};

struct Pattern : Blueprint {
	virtual void build(ConwaysCUDA::WORLD_T* world, int w_rows, int w_cols, int row, int col) const override;
	virtual int width() const override { return cols; }
	virtual int height() const override { return rows; }

	Pattern(const char* pattern, int rows, int cols)
		: pattern(pattern)
		, rows(rows)
		, cols(cols)
	{ }

	// A string of 1s and 0s, where each row is width chars long.
	const char* pattern;
	int rows, cols;
};

struct MultiPattern : Blueprint {
	virtual void build(ConwaysCUDA::WORLD_T* world, int rows, int cols, int row, int col) const override;
	virtual int width() const override;
	virtual int height() const override;

	MultiPattern(const Blueprint& pattern) 
		: blueprints{ &pattern }
		, row_offsets(1, 0)
		, col_offsets(1, 0) 
	{ }

	MultiPattern(std::initializer_list<const Blueprint*> patterns, 
			     std::initializer_list<int> row_offsets, 
			     std::initializer_list<int> col_offsets) 
		: blueprints(patterns)
		, row_offsets(row_offsets)
		, col_offsets(col_offsets) 
	{ }

	std::vector<const Blueprint*> blueprints;
	std::vector<int> row_offsets, col_offsets;
};


namespace pattern_blueprints {
	extern const Blueprint& shooter_pattern;
	extern std::vector<const Blueprint*> all_patterns;
}
