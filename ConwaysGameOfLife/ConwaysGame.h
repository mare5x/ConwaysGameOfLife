#pragma once

#include "SDL.h"

#include "Grid.h"


class ConwaysGame {
public:
	ConwaysGame(int width, int height);
	~ConwaysGame();

	void run();

	void update();

	void render();

	void input();

	void resize(int w, int h);
private:
	bool init();
	void quit();

	void handle_event(SDL_Event& e);

	void start_game();
	void stop_game();

	void restart_game();

	void render_grid_background();
	void render_grid();

	bool quit_requested;
	bool is_running;

	int w_width, w_height;
	float row_px, col_px;

	float camera_zoom;
	int camera_x, camera_y;

	int delta_milliseconds;
	Uint32 last_time;

	SDL_Renderer* renderer;
	SDL_Window* window;

	SDL_Event p_event;

	Grid grid;
};