#include "Grid.h"

Grid::Grid(int rows, int cols)
	: rows(rows), cols(cols), grid(rows * cols)
{
}

void Grid::toggle_cell(int row, int col)
{
	int idx = get_idx(row, col);
	toggle_cell(idx);
}

void Grid::toggle_cell(int idx)
{
	if (idx != -1)
		grid[idx] = !grid[idx];
}

void Grid::set_cell(int row, int col, bool val)
{
	int idx = get_idx(row, col);
	set_cell(idx, val);
}

void Grid::set_cell(int idx, bool val)
{
	if (idx != -1)
		grid[idx] = val;
}

bool Grid::get_cell(int row, int col) const
{
	int idx = get_idx(row, col);
	return get_cell(idx);
}

bool Grid::get_cell(int idx) const
{
	return idx == -1 ? false : grid[idx];
}

int Grid::get_neighbours(int row, int col) const
{
	int neighbours = 0;

	if (get_cell(row - 1, col - 1)) neighbours++;
	if (get_cell(row, col - 1)) neighbours++;
	if (get_cell(row + 1, col - 1)) neighbours++;

	if (get_cell(row - 1, col)) neighbours++;
	if (get_cell(row + 1, col)) neighbours++;

	if (get_cell(row - 1, col + 1)) neighbours++;
	if (get_cell(row, col + 1)) neighbours++;
	if (get_cell(row + 1, col + 1)) neighbours++;

	return neighbours;
}

void Grid::clear()
{
	for (size_t i = 0; i < grid.size(); i++)
		grid[i] = false;
}

int Grid::get_idx(int row, int col) const
{
	if (row >= 0 && row < rows && col >= 0 && col < cols) {
		return row * cols + col;
	}
	return -1;
}
