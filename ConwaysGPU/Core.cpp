#include "Core.h"
#include "ConwaysCUDA.h"

const int ROWS = 32;
const int COLS = 32;

const int WIDTH = 800;
const int HEIGHT = 600;

const int FPS = 60;
const float FPS_ms = 1 / static_cast<float>(FPS) * 1000;


Core::Core() :
	width(WIDTH), height(HEIGHT),
	renderer(), quit_requested(false)
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
	ConwaysCUDA::tick();
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

		window = SDL_CreateWindow("Conway's GPU", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_OPENGL);
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

	renderer.init(ROWS, COLS);

	return true;
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
	if (e.type == SDL_WINDOWEVENT) {
		if (e.window.event == SDL_WINDOWEVENT_RESIZED)
			on_resize(e.window.data1, e.window.data2);
	}
}

void Core::on_resize(int w, int h)
{
	glViewport(0, 0, w, h);
	width = w;
	height = h;

	renderer.on_resize(w, h);
}