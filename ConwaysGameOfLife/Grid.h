#pragma once

#include <vector>


class Grid {
public:
	Grid(int rows, int cols);
	
	int get_idx(int row, int col) const;

	void toggle_cell(int row, int col);
	void toggle_cell(int idx);

	void set_cell(int row, int col, bool val);
	void set_cell(int idx, bool val);

	bool get_cell(int row, int col) const;
	bool get_cell(int idx) const;

	int get_neighbours(int row, int col) const;

	void clear();
	size_t size() { return grid.size(); }
private:
	int rows, cols;

	std::vector<bool> grid;
};