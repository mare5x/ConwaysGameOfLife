#include "Core.h"
#include "ConwaysCUDA.h"
#include <algorithm>
#include <ctime>
#include <cstdlib>

const int ROWS = 1024; // 2048
const int COLS = 1024; // 2048

const vec2<float> base_camera_center(0.5f, 0.5f);
	
const int WIDTH = 800;
const int HEIGHT = 600;

const int FPS = 60;
const float FPS_ms = 1 / static_cast<float>(FPS) * 1000;


template<class T>
const T& clamp(const T& val, const T& lo, const T& hi)
{
	if (val > hi) return hi;
	if (val < lo) return lo;
	return val;
}


Core::Core() :
	width(WIDTH), height(HEIGHT),
	renderer(), camera_center(0.5f, 0.5f)
{
	if (!init())
		quit();
}

void Core::run()
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

void Core::input()
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

void Core::update()
{
	static unsigned int prev_tick = 0;
	static int accumulator = 0;
	const int dt = 1000 / ticks_per_second;
	if (is_playing) {
		unsigned int ticks = SDL_GetTicks();
		accumulator += ticks - prev_tick;
		while (accumulator > dt) {
			ConwaysCUDA::tick();
			++generation;
			accumulator -= dt;
		}
	}
	prev_tick = SDL_GetTicks();
}

void Core::render()
{
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	renderer.render();

	SDL_GL_SwapWindow(window);
}

bool Core::init()
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

bool Core::init_gl()
{
	SDL_GL_SetSwapInterval(0);  // uncap FPS

	on_resize(width, height);

	renderer.init(ROWS, COLS, initial_world_state.data());
	renderer.on_resize(width, height);
	renderer.set_zoom(zoom);
	renderer.set_camera_center(camera_center.x, camera_center.y);

	return true;
}

void Core::init_world_state()
{
	initial_world_state.resize(ROWS * COLS);
	//for (int row = 0; row < ROWS; ++row) {
	//	for (int col = 0; col < COLS; ++col) {
	//		if ((row + col) % 4 == 0)
	//			initial_world_state[row * COLS + col] = 1.0f;
	//	}
	//}
}

void Core::randomize_world()
{
	set_is_playing(false);
	std::srand(std::time(nullptr));
	for (int i = 0; i < initial_world_state.size(); ++i)
		initial_world_state[i] = (std::rand() % 2 == 0) ? 1.0f : 0.0f;
	renderer.set_world_grid(initial_world_state.data());
}

void Core::set_is_playing(bool val)
{
	is_playing = val;
	if (is_playing) {
		renderer.set_world_grid(initial_world_state.data());
		generation = 0;
	}
}

void Core::toggle_tile_state(int row, int col)
{
	if (col >= 0 && col < COLS && row >= 0 && row < ROWS) {
		float& state = initial_world_state[row * COLS + col];
		state = (state > 0 ? 0 : 1);
		set_is_playing(false);
		renderer.set_world_grid(initial_world_state.data());
	}
}

vec2<float> Core::screen_to_world(int x, int y)
{
	y = height - y;  // screen coordinate (0,0) is top left
	float edge = std::min(width, height) * zoom;
	vec2<float> cam_offset = (base_camera_center - camera_center) * edge;
	vec2<float> offset = (vec2<float>(width, height) - edge) / 2.0f + cam_offset;
	return (vec2<float>(x, y) - offset) / edge;
}

void Core::move_camera_center(float dx, float dy)
{
	camera_center.x += dx;
	camera_center.y += dy;
	renderer.set_camera_center(camera_center.x, camera_center.y);
}

void Core::update_zoom(int sign)
{
	// Zoom as if travelling on the exponential function.
	static float zoom_x = 0.0f;

	// Cool way to get the sign (- or +) of a number.
	sign = (sign > 0) - (sign < 0);

	zoom_x = clamp(zoom_x + 0.1f * sign, -1.0f, 5.0f);
	zoom = std::exp(zoom_x);
	renderer.set_zoom(zoom);
}

void Core::print_stats()
{
	printf("---- INFO ----\n");
	printf("Window: %d px x %d px\n", width, height);
	printf("Zoom: %.3f\n", zoom);
	printf("Camera center: (%.3f, %.3f)\n", camera_center.x, camera_center.y);
	printf("Ticks per second: %.2f\n", ticks_per_second);
	printf("%d rows by %d columns\n", ROWS, COLS);
	printf("On generation %d\n\n", generation);
}

void Core::quit()
{
	ConwaysCUDA::exit();

	SDL_GL_DeleteContext(context);
	context = nullptr;
	SDL_DestroyWindow(window);
	window = nullptr;

	SDL_Quit();
}

void Core::handle_input(SDL_Event & e)
{
	static vec2<float> cam_delta;

	if (e.type == SDL_WINDOWEVENT) {
		if (e.window.event == SDL_WINDOWEVENT_RESIZED)
			on_resize(e.window.data1, e.window.data2);
	}
	else if (e.type == SDL_MOUSEWHEEL) {
		update_zoom(e.wheel.y);
	}
	else if (e.type == SDL_MOUSEBUTTONUP) {
		vec2<float> world_pos = screen_to_world(e.button.x, e.button.y);
		int row = world_pos.y * ROWS;
		int col = world_pos.x * COLS;
		toggle_tile_state(row, col);
		printf("%d %d (%.2f %.2f)\n", row, col, world_pos.x, world_pos.y);
	}
	else if (e.type == SDL_KEYDOWN) {
		switch (e.key.keysym.sym) {
			case SDLK_q:
				ticks_per_second = clamp(ticks_per_second - 0.25f, 0.25f, 420.0f);
				break;
			case SDLK_e:
				ticks_per_second = clamp(ticks_per_second + 0.25f, 0.25f, 420.0f);
				break;
			case SDLK_w: cam_delta.y =  0.01f / zoom; break;
			case SDLK_s: cam_delta.y = -0.01f / zoom; break;
			case SDLK_d: cam_delta.x =  0.01f / zoom; break;
			case SDLK_a: cam_delta.x = -0.01f / zoom; break;
		}
	}
	else if (e.type == SDL_KEYUP) {
		switch (e.key.keysym.sym) {
			case SDLK_SPACE: set_is_playing(!is_playing); break;
			case SDLK_p: print_stats(); break;
			case SDLK_r: randomize_world(); break;

			case SDLK_w: cam_delta.y = 0; break;
			case SDLK_s: cam_delta.y = 0; break;
			case SDLK_d: cam_delta.x = 0; break;
			case SDLK_a: cam_delta.x = 0; break;
		}
	}
	
	if (std::abs(cam_delta.x) > 1e-6 || std::abs(cam_delta.y) > 1e-6)
		move_camera_center(cam_delta.x, cam_delta.y);
}

void Core::on_resize(int w, int h)
{
	glViewport(0, 0, w, h);
	width = w;
	height = h;

	renderer.on_resize(w, h);
}