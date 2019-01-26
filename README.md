# Conway's Game Of Life

## About

_ConwaysGameOfLife_ is an old SDL2 only implementation (2017).  
_ConwaysGPU_ is the newer GPU based implementation (2019).  

**Conway's Game of Life** implemented using _**NVIDIA**'s **CUDA**_ and modern _**OpenGL**_. _**SDL2**_ is used for window management.

A _live_ cell is drawn red. The bluish colors of a certain cell represent the number of neighbours that cell had in the _previous_ generation. By default the world is 1024 x 1024 cells large (change _ROWS_ and _COLS_ constants in _ConwaysGameOfLife.cpp_).

Cells are drawn using instanced drawing of a quad whose position is calculated in the vertex shader based on the index of the current instance. The grid and other effects are all done right in the vertex and fragment shaders.

The cell state is given to the vertex/fragment shader in a 1 dimensional array of integers. That is the array (_VBO_) the CUDA code works with. The GPU is responsible for simulating one _tick_ of the Game by updating the VBO containing the states of each cell. 

Tested on a _NVIDIA GeForce GTX 1060 (6 GB)_ GPU with _Windows 10 (64-bit)_. Compiled with _Microsoft Visual Studio Community 2017_.

### Usage
 * Mouse click on a cell to toggle it's state.
 * Mouse wheel to zoom.
 * WASD keys to move around.
 * SPACE to start/stop the simulation.
 * Q and E to decrease/increase the tick rate.
 * R to randomize the world.
 * C to clear the world state.
 * G to show/hide grid lines.
 * P to print some stats to the console.