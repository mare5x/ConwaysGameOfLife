#pragma once
#include "SDL.h"
#include "glad/glad.h"
#include "Renderer.h"
#include <vector>
#include "vec2.h"
#include "pattern_blueprints.h"


class ConwaysGameOfLife {
public:
	ConwaysGameOfLife();
	~ConwaysGameOfLife() { quit(); }

	static const int ROWS = 1024; // 2048
	static const int COLS = 1024; // 2048

	void run();

	void input();
	void update();
	void render();
private:
	bool init();
	bool init_gl();
	void quit();

	void on_resize(int w, int h);

	void handle_input(SDL_Event& e);

	void init_world_state(bool send_to_gpu = false);
	void randomize_world();
	void world_to_renderer();
	void reset_world();

	void toggle_tile_state(int x, int y);
	bool tile_in_bounds(int row, int col) const { return row >= 0 && row < ROWS && col >= 0 && col < COLS; }

	// Convert (x, y) in screen coordinates to world coordinates [0, 1].
	vec2<float> screen_to_world(int x, int y);
	// x == rows, y == cols
	vec2<int> screen_to_tile(int x, int y);

	void move_camera_center(float dx, float dy);
	void update_zoom(int sign);

	void update_tick_rate(int sign);

	// Place the currently set pattern so that the top left corner is at 
	// (x,y) in screen coordinates.
	void place_pattern(int x, int y);
	void set_blueprint(const Blueprint& blueprint, int row, int col) {
		blueprint.build(initial_world_state.data(), ROWS, COLS, row, col);
	}
	void toggle_pattern_hovering();
	void update_pattern_hover(int x, int y, bool force = false);
	void update_pattern_hover(bool force = false);
	void next_pattern();

	void print_stats();

	int width, height;

	bool quit_requested = false;

	Renderer renderer;

	SDL_Event cur_event;

	SDL_Window* window;
	SDL_GLContext context;

	// --- Game specifics --- //
	bool is_playing = false;
	bool is_grid_visible = true;
	bool is_pattern_hovering = false;
	unsigned int generation = 0;

	vec2<float> camera_velocity;
	vec2<float> camera_center;
	float zoom = 1.0f;
	float ticks_per_second = 10.0f;

	int current_blueprint = 0;
	int pattern_blueprint = 0;
	vec2<int> pattern_tile;

	// We can't just use an std::array or something similar because 
	// it allocates memory on the stack. That causes crashes when
	// the grid size exceeds a certain amount (stack size limit). 
	std::vector<ConwaysCUDA::WORLD_T> initial_world_state;
};