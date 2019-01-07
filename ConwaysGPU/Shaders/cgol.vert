#version 330 core
layout (location = 0) in vec2 norm_pos;
layout (location = 1) in float state_enabled;

uniform int rows;
uniform int cols;

out float enabled;
out vec2 grid_pos;

void main()
{
	int row = gl_InstanceID / cols;
	int col = gl_InstanceID % cols;
	grid_pos.x = col / float(cols);
	grid_pos.y = row / float(rows);
	enabled = state_enabled;
	float x_size = 2.0 / cols;
	float y_size = 2.0 / rows;
	float zoom = 2.0;
	gl_Position = vec4(grid_pos.x * zoom - zoom / 2 + zoom *  norm_pos.x * x_size, 
					   grid_pos.y * zoom - zoom / 2 + zoom * norm_pos.y * y_size, 
					   0.0, 1.0);
}