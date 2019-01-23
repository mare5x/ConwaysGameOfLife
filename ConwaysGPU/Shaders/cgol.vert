#version 330 core
layout (location = 0) in vec2 norm_pos;
layout (location = 1) in float state_enabled;

uniform vec2 screen_size;
uniform int rows;
uniform int cols;
uniform float zoom;
uniform vec2 camera_center;  // in [0, 1] coordinates

out float enabled;

const vec2 base_camera_center = vec2(0.5, 0.5);

// Work with coordinates in the range [0, 1] and then transform
// them to OpenGL space with this function ([-1, 1]).
vec2 norm_to_vertex(vec2 pos)
{
	// Fit everything into a square of size edge centered on screen.

	vec2 cam_offset = base_camera_center - camera_center;
	pos += cam_offset;

	float edge = zoom * min(screen_size.x, screen_size.y);
	vec2 edge_norm = edge / screen_size * 2.0;  // component-wise operations
	return pos * edge_norm - edge_norm / 2.0;
}

void main()
{
	enabled = state_enabled;  // Tell the fragment shader about the state of this tile.

	// Calculate the row and column information from the instance index of the current instanced draw command.
	int row = gl_InstanceID / cols;
	int col = gl_InstanceID % cols;

	vec2 grid_pos;
	grid_pos.x = col / float(cols);
	grid_pos.y = row / float(rows);

	vec2 norm_quad_size = vec2(1.0 / cols, 1.0 / rows);
	vec2 vert_offset = norm_pos * norm_quad_size;

	// Vertex position in [0,1] coordinates.
	grid_pos += vert_offset;

	gl_Position = vec4(norm_to_vertex(grid_pos), 0.0, 1.0);
}