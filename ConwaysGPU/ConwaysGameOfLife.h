#pragma once
#include "SDL.h"
#include "glad/glad.h"
#include "Renderer.h"
#include <vector>
#include "vec2.h"


class ConwaysGameOfLife {
public:
	ConwaysGameOfLife();
	~ConwaysGameOfLife() { quit(); }

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

	void set_is_playing(bool val);
	void toggle_tile_state(int x, int y);

	// Convert (x, y) in screen coordinates to world coordinates [0, 1].
	vec2<float> screen_to_world(int x, int y);

	void move_camera_center(float dx, float dy);
	void update_zoom(int sign);

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
	unsigned int generation = 0;

	vec2<float> camera_velocity;
	vec2<float> camera_center;
	float zoom = 1.0f;
	float ticks_per_second = 10.0f;

	// We can't just use an std::array or something similar because 
	// it allocates memory on the stack. That causes crashes when
	// the grid size exceeds a certain amount (stack size limit). 
	std::vector<GLbyte> initial_world_state;
};