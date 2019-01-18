#version 330 core
layout (location = 0) in vec2 norm_pos;
layout (location = 1) in float state_enabled;

uniform int rows;
uniform int cols;

out float enabled;
out vec2 grid_pos;

void main()
{
	enabled = state_enabled;  // Tell the fragment shader about the state of this tile.

	float aspect_ratio = 800.0 / 600.0;

	// Calculate the row and column information from the instance index of the current instanced draw command.
	int row = gl_InstanceID / cols;
	int col = gl_InstanceID % cols;

	grid_pos.x = col / float(cols);
	grid_pos.y = row / float(rows);

	float x_size = 1.0 / cols;
	float y_size = 1.0 / rows;
	vec2 vert_offset = vec2(norm_pos.x * x_size, norm_pos.y * y_size);

	// Vertex position in [0,1] coordinates.
	grid_pos += vert_offset;
	grid_pos.y *= aspect_ratio;

	float zoom = 1.0;
	gl_Position = vec4(zoom * (2.0 * grid_pos.x - 1.0), 
					   zoom * (2.0 * grid_pos.y - 1.0), 
					   0.0, 1.0);
}