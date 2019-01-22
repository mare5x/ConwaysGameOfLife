#pragma once
#include "SDL.h"
#include "glad/glad.h"
#include "Renderer.h"
#include <array>


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
	void toggle_tile_state(int row, int col);
	struct Tile screen_to_tile(int x, int y);

	int width, height;

	bool quit_requested;

	Renderer renderer;

	SDL_Event cur_event;

	SDL_Window* window;
	SDL_GLContext context;

	// --- Game specifics --- //
	const static int ROWS = 32;
	const static int COLS = 32;
	
	float zoom;
	bool is_playing;

	std::array<float, ROWS * COLS> initial_world_state;
};