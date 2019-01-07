#include "ConwaysGame.h"

const int ROWS = 64;
const int COLS = 64;


ConwaysGame::ConwaysGame(int width, int height)
	: p_event(), quit_requested(false), grid(ROWS, COLS), w_width(width), w_height(height), delta_milliseconds(200),
	camera_zoom(1), camera_x(0), camera_y(0)
{
	init();
	resize(width, height);
	last_time = SDL_GetTicks();
}

ConwaysGame::~ConwaysGame()
{
	quit();
}

void ConwaysGame::run()
{
	while (!quit_requested) {
		input();

		SDL_SetRenderDrawColor(renderer, 0xff, 0xff, 0xff, 0xff);
		SDL_RenderClear(renderer);

		update();

		render();
	}
}

void ConwaysGame::update()
{
	if (!is_running)
		return;

	if (SDL_GetTicks() - last_time >= delta_milliseconds) {
		last_time = SDL_GetTicks();
	} else
		return;

	// Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
	// Any live cell with two or three live neighbours lives on to the next generation.
	// Any live cell with more than three live neighbours dies, as if by overpopulation.
	// Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

	std::vector<int> schedule_death;
	std::vector<int> schedule_life;

	int neighbours = 0;
	for (int row = 0; row < ROWS; row++) {
		for (int col = 0; col < COLS; col++) {
			neighbours = grid.get_neighbours(row, col);
			if (grid.get_cell(row, col)) {
				if (neighbours < 2 || neighbours > 3)
					schedule_death.push_back(grid.get_idx(row, col));
			} else {
				if (neighbours == 3)
					schedule_life.push_back(grid.get_idx(row, col));
			}
		}
	}

	for (int idx : schedule_death)
		grid.set_cell(idx, false);
	for (int idx : schedule_life)
		grid.set_cell(idx, true);
}

void ConwaysGame::render()
{
	render_grid_background();
	render_grid();

	SDL_RenderPresent(renderer);
}

void ConwaysGame::input()
{
	while (SDL_PollEvent(&p_event)) {
		if (p_event.type == SDL_QUIT)
			quit_requested = true;
		else {
			handle_event(p_event);
		}
	}
}

void ConwaysGame::resize(int w, int h)
{
	w_width = w;
	w_height = h;

	float grid_aspect_ratio = (w * ROWS) / static_cast<float>(COLS * h);

	col_px = w / static_cast<float>(COLS);
	row_px = h / static_cast<float>(ROWS) * grid_aspect_ratio;
}

bool ConwaysGame::init()
{
	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
		return false;
	}
	else {
		window = SDL_CreateWindow("Conway's Game of Life", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w_width, w_height, SDL_WINDOW_SHOWN);
		if (window == NULL) {
			SDL_Log("Couldn't create window: %s", SDL_GetError());
			return false;
		}

		renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_TARGETTEXTURE | SDL_RENDERER_PRESENTVSYNC);
		if (renderer == NULL) {
			SDL_Log("Couldn't create renderer: %s", SDL_GetError());
			return false;
		}
	}
	return true;
}

void ConwaysGame::quit()
{
	SDL_Quit();

	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	renderer = nullptr;
	window = nullptr;
}

void ConwaysGame::handle_event(SDL_Event& e)
{
	if (e.type == SDL_KEYUP) {
		switch (e.key.keysym.sym) {
		case SDLK_SPACE:
			if (is_running)
				stop_game();
			else
				start_game();
			break;
		case SDLK_r:
			restart_game();
			break;
		default:
			break;
		}
	} else if (e.type == SDL_KEYDOWN) {
		switch (e.key.keysym.sym) {
		case SDLK_q:
			delta_milliseconds += 25;
			break;
		case SDLK_e:
			delta_milliseconds -= 25;
			if (delta_milliseconds <= 0)
				delta_milliseconds = 25;
			break;
		case SDLK_LEFT:
			camera_x += 5;
			break;
		case SDLK_RIGHT:
			camera_x -= 5;
			break;
		case SDLK_DOWN:
			camera_y -= 5;
			break;
		case SDLK_UP:
			camera_y += 5;
			break;
		default:
			break;
		}
	} else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {
		int row = (e.button.y - camera_y) / row_px / camera_zoom;
		int col = (e.button.x - camera_x) / col_px / camera_zoom;
		grid.toggle_cell(row, col);
	} else if (e.type == SDL_MOUSEWHEEL) {
		if (e.wheel.y == 1)
			camera_zoom += 0.1f;
		else if (e.wheel.y == -1)
			camera_zoom -= 0.1f;
	}
}

void ConwaysGame::start_game()
{
	is_running = true;
}

void ConwaysGame::stop_game()
{
	is_running = false;
}

void ConwaysGame::restart_game()
{
	is_running = false;
	grid.clear();
}

void ConwaysGame::render_grid_background()
{
	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0xff);  // black
	for (int row = 0; row <= ROWS; row++) {
		SDL_RenderDrawLine(renderer, 0, camera_y + row * row_px * camera_zoom, w_width, camera_y + row * row_px * camera_zoom);
	}
	for (int col = 0; col <= COLS; col++) {
		SDL_RenderDrawLine(renderer, camera_x + col * col_px * camera_zoom, 0, camera_x + col * col_px * camera_zoom, w_height);
	}
}

void ConwaysGame::render_grid()
{
	SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0xff);  // black
	SDL_Rect rect;
	rect.w = std::ceil(col_px * camera_zoom);
	rect.h = std::ceil(row_px * camera_zoom);
	for (int row = 0; row < ROWS; row++) {
		for (int col = 0; col < COLS; col++) {
			if (!grid.get_cell(row, col))
				continue;

			rect.x = camera_x + col * col_px * camera_zoom;
			rect.y = camera_y + row * row_px * camera_zoom;

			SDL_RenderFillRect(renderer, &rect);
		}
	}
}
