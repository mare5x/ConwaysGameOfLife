#version 330 core

uniform vec2 screen_size;
uniform int rows;
uniform int cols;
uniform float zoom;
uniform vec2 camera_center;  // in [0, 1] coordinates

flat in int tile_state;

out vec4 frag;

const vec2 base_camera_center = vec2(0.5, 0.5);
const float MAX_GRID_K = 16;

bool point_on_grid(vec2 screen_pos, int parts, int width)
{
	// Use modulo arithmetic to determine if the given point is on the grid.
	// width is in pixels

	float base_edge = min(screen_size.x, screen_size.y);
	float edge = zoom * base_edge;

	// How many times do we have to split the grid in half
	// so that the distance between two grid lines is at most
	// base_edge/MAX_GRID_K apart (which is a constant)?
	// Solve for x (:= max_parts):
	// \frac{edge}{2^x} = \frac{base_edge}{MAX_GRID_K}

	int max_parts = int(log2(MAX_GRID_K * edge / base_edge));

	// wrap_amount is basically the distance between the grid lines in pixels

	float wrap_amount = max(edge / exp2(max_parts), edge / float(parts));
	float radius = width / 2.0;

	vec2 cam_offset = base_camera_center - camera_center;
	cam_offset *= edge;
	vec2 offset = (screen_size - edge) / 2.0 + cam_offset;

	vec2 res = mod(screen_pos - offset, wrap_amount); 
	return res.x <= radius || wrap_amount - res.x <= radius || 
		   res.y <= radius || wrap_amount - res.y <= radius;
}

bool screen_grid(vec2 screen_pos, int parts)
{
	return point_on_grid(screen_pos, parts, 1) || 
	  	   point_on_grid(screen_pos, parts / 16, 2);
}

void main()
{
	// ALWAYS WATCH OUT FOR TYPE CONVERSION WHEN WORKING WITH FLOATS AND INTS!!!

	// NOTE:
	// By default, gl_FragCoord assumes a lower-left origin for window 
	// coordinates and assumes pixel centers are located at half-pixel centers.

	float aspect_ratio = screen_size.x / screen_size.y;
	vec4 pos = gl_FragCoord;

	if (screen_grid(pos.xy, rows))
		frag = vec4(1, 1, 1, 0.3);

	if (tile_state > 0)
		frag = vec4(1, 0, 0, 1);
}