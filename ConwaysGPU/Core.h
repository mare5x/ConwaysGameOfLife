#pragma once
#include "SDL.h"
#include "glad/glad.h"
#include "Renderer.h"


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

	int width, height;

	bool quit_requested;

	Renderer renderer;

	SDL_Event cur_event;

	SDL_Window* window;
	SDL_GLContext context;
};