#include "ConwaysGameOfLife.h"
#include "ConwaysCUDA.h"
#include <algorithm>
#include <ctime>
#include <cstdlib>

using ConwaysCUDA::WORLD_T;

const vec2<float> base_camera_center(0.5f, 0.5f);
	
const int WIDTH = 800;
const int HEIGHT = 600;

const int FPS = 60;
const int FPS_ms = 1000 / FPS;


template<class T>
const T& clamp(const T& val, const T& lo, const T& hi)
{
	if (val > hi) return hi;
	if (val < lo) return lo;
	return val;
}


ConwaysGameOfLife::ConwaysGameOfLife() 
	: width(WIDTH)
	, height(HEIGHT)
	, renderer()
	, camera_center(0.5f, 0.5f)
	, camera_velocity()
	, pattern_tile()
{
	if (!init())
		quit();
}

void ConwaysGameOfLife::run()
{
	while (!quit_requested) {
		unsigned int start_time = SDL_GetTicks();

		input();

		update();

		render();

		unsigned int frame_time = SDL_GetTicks() - start_time;
		if (frame_time < FPS_ms)
			SDL_Delay(FPS_ms - frame_time);
	}
}

void ConwaysGameOfLife::input()
{
	while (SDL_PollEvent(&cur_event)) {
		if (cur_event.type == SDL_QUIT) {
			quit_requested = true;
		}
		else {
			handle_input(cur_event);
		}
	}
}

void ConwaysGameOfLife::update()
{
	static unsigned int prev_ticks = 0;
	static int accumulator = 0;
	const int dt = 1000 / ticks_per_second;

	ConwaysCUDA::start_interop();

	if (std::abs(camera_velocity.x) > 1e-6 || std::abs(camera_velocity.y) > 1e-6)
		move_camera_center(camera_velocity.x / zoom, camera_velocity.y / zoom);

	// Handle mouse hover patterns.
	if (is_pattern_hovering) {
		int mouse_x, mouse_y;
		SDL_GetMouseState(&mouse_x, &mouse_y);
		handle_pattern_hover(mouse_x, mouse_y);
	}

	if (is_playing) {
		unsigned int ticks = SDL_GetTicks();
		accumulator += ticks - prev_ticks;
		while (accumulator > dt) {
			ConwaysCUDA::tick();
			++generation;
			accumulator -= dt;
		}
	}
	prev_ticks = SDL_GetTicks();

	ConwaysCUDA::stop_interop();
}

void ConwaysGameOfLife::render()
{
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	renderer.render();

	SDL_GL_SwapWindow(window);
}

bool ConwaysGameOfLife::init()
{
	if (SDL_Init(SDL_INIT_VIDEO) == 0) {
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

		window = SDL_CreateWindow("Conway's GPU", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
								  width, height, 
								  SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
		if (!window) {
			SDL_Log("CREATE WINDOW ERROR: %s", SDL_GetError());
			return false;
		}

		context = SDL_GL_CreateContext(window);
		if (!context) {
			SDL_Log("CREATE CONTEXT ERROR: %s", SDL_GetError());
			return false;
		}

		if (!gladLoadGLLoader(static_cast<GLADloadproc>(SDL_GL_GetProcAddress)))
			return false;

		init_world_state();

		return init_gl();
	}
	else {
		SDL_Log("INIT ERROR: %s", SDL_GetError());
		return false;
	}
}

bool ConwaysGameOfLife::init_gl()
{
	SDL_GL_SetSwapInterval(0);  // uncap FPS
	
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	on_resize(width, height);

	renderer.init(ROWS, COLS);
	renderer.on_resize(width, height);
	renderer.set_zoom(zoom);
	renderer.set_camera_center(camera_center.x, camera_center.y);
	renderer.set_world_grid(initial_world_state.data());

	return true;
}

void ConwaysGameOfLife::init_world_state(bool send_to_gpu)
{
	initial_world_state.resize(ROWS * COLS);
	std::fill(initial_world_state.begin(), initial_world_state.end(), 0);

	if (send_to_gpu)
		world_to_renderer();

	is_playing = false;
}

void ConwaysGameOfLife::randomize_world()
{
	std::srand(std::time(nullptr));
	for (int i = 0; i < initial_world_state.size(); ++i)
		initial_world_state[i] = (std::rand() % 4 == 0) ? 1 : 0;
	world_to_renderer();
}

void ConwaysGameOfLife::world_to_renderer()
{
	set_is_playing(false);
	renderer.set_world_grid(initial_world_state.data());
}

void ConwaysGameOfLife::set_is_playing(bool val)
{
	is_playing = val;
	if (is_playing) {
		renderer.set_world_grid(initial_world_state.data());
		generation = 0;
	}
}

void ConwaysGameOfLife::toggle_tile_state(int x, int y)
{
	vec2<int> tile = screen_to_tile(x, y);
	int row = tile.x;
	int col = tile.y;
	if (tile_in_bounds(row, col)) {
		if (is_playing) {
			ConwaysCUDA::start_interop();
			ConwaysCUDA::toggle_cell(row, col);
			ConwaysCUDA::stop_interop();
			//renderer.toggle_cell(row, col);
		}
		else {
			WORLD_T& state = initial_world_state[row * COLS + col];
			state = (state > 0 ? 0 : 1);
			world_to_renderer();
		}
	}
}

vec2<float> ConwaysGameOfLife::screen_to_world(int x, int y)
{
	// NOTE: screen coordinate (0,0) is top left, but world (0,0) is bottom left.
	y = height - y; 
	float edge = std::min(width, height) * zoom;
	vec2<float> cam_offset = (base_camera_center - camera_center) * edge;
	vec2<float> offset = (vec2<float>(width, height) - edge) / 2.0f + cam_offset;
	return (vec2<float>(x, y) - offset) / edge;
}

vec2<int> ConwaysGameOfLife::screen_to_tile(int x, int y)
{
	vec2<float> world_pos = screen_to_world(x, y);
	int row = world_pos.y * ROWS;
	int col = world_pos.x * COLS;
	return { row, col };
}

void ConwaysGameOfLife::move_camera_center(float dx, float dy)
{
	camera_center.x += dx;
	camera_center.y += dy;
	renderer.set_camera_center(camera_center.x, camera_center.y);
}

void ConwaysGameOfLife::update_zoom(int sign)
{
	// Zoom as if travelling on the exponential function.
	static float zoom_x = 0.0f;

	// Cool way to get the sign (- or +) of a number.
	sign = (sign > 0) - (sign < 0);

	zoom_x = clamp(zoom_x + 0.1f * sign, -1.0f, 5.0f);
	zoom = std::exp(zoom_x);
	renderer.set_zoom(zoom);
}

void ConwaysGameOfLife::place_pattern(int x, int y)
{
	const Blueprint& blueprint = *pattern_blueprints::all_patterns[current_blueprint];
	vec2<int> tile = screen_to_tile(x, y);
	if (!tile_in_bounds(tile.x, tile.y)) return;

	if (is_playing) {
		ConwaysCUDA::set_pattern(blueprint, tile.x, tile.y);
	} else {
		set_blueprint(blueprint, tile.x, tile.y);
		world_to_renderer();
	}
}

void ConwaysGameOfLife::toggle_pattern_hovering()
{
	is_pattern_hovering = !is_pattern_hovering;

	// Remove the hovering remains.
	if (!is_pattern_hovering) {
		const Blueprint& prev_blueprint = *pattern_blueprints::all_patterns[pattern_blueprint];
		ConwaysCUDA::set_hover_pattern(prev_blueprint, pattern_tile.x, pattern_tile.y, false);
	}
}

void ConwaysGameOfLife::handle_pattern_hover(int x, int y)
{
	vec2<int> tile = screen_to_tile(x, y);
	if (tile == pattern_tile || !tile_in_bounds(tile.x, tile.y)) return;

	// Remove previous pattern, then hover the current one ...
	const Blueprint& prev_blueprint = *pattern_blueprints::all_patterns[pattern_blueprint];
	ConwaysCUDA::set_hover_pattern(prev_blueprint, pattern_tile.x, pattern_tile.y, false);

	const Blueprint& blueprint = *pattern_blueprints::all_patterns[current_blueprint];
	ConwaysCUDA::set_hover_pattern(blueprint, tile.x, tile.y, true);

	pattern_tile = tile;
	pattern_blueprint = current_blueprint;
}

void ConwaysGameOfLife::print_stats()
{
	printf("---- INFO ----\n");
	printf("Window: %d px x %d px\n", width, height);
	printf("Zoom: %.3f\n", zoom);
	printf("Camera center: (%.3f, %.3f)\n", camera_center.x, camera_center.y);
	printf("Ticks per second: %.2f\n", ticks_per_second);
	printf("%d rows by %d columns\n", ROWS, COLS);
	printf("On generation %d\n\n", generation);
}

void ConwaysGameOfLife::quit()
{
	ConwaysCUDA::exit();

	SDL_GL_DeleteContext(context);
	context = nullptr;
	SDL_DestroyWindow(window);
	window = nullptr;

	SDL_Quit();
}

void ConwaysGameOfLife::handle_input(SDL_Event & e)
{
	switch (e.type) {
	case SDL_WINDOWEVENT:
		switch (e.window.event) {
		case SDL_WINDOWEVENT_RESIZED:
			on_resize(e.window.data1, e.window.data2);
		}
		break;
	case SDL_MOUSEWHEEL: update_zoom(e.wheel.y); break;
	case SDL_MOUSEBUTTONUP: 
		switch (e.button.button) {
		case SDL_BUTTON_LEFT: 
			if (is_pattern_hovering)
				place_pattern(e.button.x, e.button.y);
			else
				toggle_tile_state(e.button.x, e.button.y); 
			break;
		case SDL_BUTTON_RIGHT: place_pattern(e.button.x, e.button.y); break;
		}
		break;
	case SDL_KEYDOWN:
		switch (e.key.keysym.sym) {
			case SDLK_q:
				ticks_per_second = clamp(ticks_per_second - 0.25f, 0.25f, 420.0f);
				break;
			case SDLK_e:
				ticks_per_second = clamp(ticks_per_second + 0.25f, 0.25f, 420.0f);
				break;
			case SDLK_w: camera_velocity.y =  0.01f; break;
			case SDLK_s: camera_velocity.y = -0.01f; break;
			case SDLK_d: camera_velocity.x =  0.01f; break;
			case SDLK_a: camera_velocity.x = -0.01f; break;
		}
		break;
	case SDL_KEYUP:
		switch (e.key.keysym.sym) {
			case SDLK_SPACE: set_is_playing(!is_playing); break;
			case SDLK_p: print_stats(); break;
			case SDLK_r: randomize_world(); break;
			case SDLK_c: init_world_state(true); break;
			case SDLK_x: world_to_renderer(); break;
			case SDLK_g: renderer.set_grid_visibility(is_grid_visible = !is_grid_visible); break;

			case SDLK_w: camera_velocity.y = 0; break;
			case SDLK_s: camera_velocity.y = 0; break;
			case SDLK_d: camera_velocity.x = 0; break;
			case SDLK_a: camera_velocity.x = 0; break;

			case SDLK_1: 
				current_blueprint = 
					(current_blueprint + 1) % pattern_blueprints::all_patterns.size(); 
				break;
			case SDLK_TAB: toggle_pattern_hovering(); break;
		}
		break;
	}
}

void ConwaysGameOfLife::on_resize(int w, int h)
{
	glViewport(0, 0, w, h);
	width = w;
	height = h;

	renderer.on_resize(w, h);
}