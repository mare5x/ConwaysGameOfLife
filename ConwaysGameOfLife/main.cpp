#include "ConwaysGame.h"


const int WIDTH = 800;
const int HEIGHT = 600;


int main(int argc, char* argv[])
{
	ConwaysGame game(WIDTH, HEIGHT);
	game.run();

	return 0;
}