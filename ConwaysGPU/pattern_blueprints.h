#pragma once
#include <vector>
#include "ConwaysCUDA.h"

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

	// FIXME: workaround for CUDA ... 
	virtual BlueprintType type() const = 0;
};

struct Pattern : Blueprint {
	virtual void build(ConwaysCUDA::WORLD_T* world, int w_rows, int w_cols, int row, int col) const override;
	virtual int width() const override { return cols; }
	virtual int height() const override { return rows; }

	virtual BlueprintType type() const override { return BlueprintType::Pattern; }

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

	virtual BlueprintType type() const override { return BlueprintType::MultiPattern; }

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
	enum PatternCategory {
		STILL_LIFES, OSCILLATORS, SPACESHIPS, OTHERS, _N_PATTERN_CATEGORIES
	};
	extern std::vector<std::vector<const Blueprint*> > all_patterns;

	// Patterns based on https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
	extern std::vector<const Blueprint*>& still_lifes;
	extern std::vector<const Blueprint*>& oscillators;
	extern std::vector<const Blueprint*>& spaceships;
	extern std::vector<const Blueprint*>& others;
}
