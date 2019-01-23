#pragma once
#include "SDL.h"
#include "glad/glad.h"
#include "Renderer.h"
#include <vector>


class Core {
public:
	Core();
	~Core() { quit(); }

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

	void init_world_state();
	void randomize_world();

	void set_is_playing(bool val);
	void toggle_tile_state(int row, int col);
	struct Tile screen_to_tile(int x, int y);

	void print_stats();

	int width, height;

	bool quit_requested = false;

	Renderer renderer;

	SDL_Event cur_event;

	SDL_Window* window;
	SDL_GLContext context;

	// --- Game specifics --- //
	const static int ROWS = 2048; // 1024;
	const static int COLS = 2048; // 1024;
	
	bool is_playing = false;
	unsigned int generation = 0;

	float zoom = 1.0f;
	float ticks_per_second = 2.0f;

	// We can't just use an std::array or something similar because 
	// it allocates memory on the stack. That causes crashes when
	// the grid size exceeds a certain amount (stack size limit). 
	std::vector<float> initial_world_state;
};