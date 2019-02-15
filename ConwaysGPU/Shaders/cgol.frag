#version 330 core

uniform vec2 screen_size;
uniform int rows;
uniform int cols;
uniform float zoom;
uniform vec2 camera_center;  // in [0, 1] coordinates
uniform bool show_grid;

flat in int tile_state;
flat in int tile_age;

out vec4 frag;

const vec2 base_camera_center = vec2(0.5, 0.5);

#define PI						3.14159265
#define MAX_GRID_K				32.0		// higher number -> more grid tiles
#define BREATHE_PERIOD		    128.0		// in ticks
#define FADE_AWAY_PERIOD        32.0		// in ticks
#define FADE_AWAY_THRESHOLD     0.2			// min scaling factor (alpha)
#define FADE_AWAY_THRESHOLD_X   (pow(1 / FADE_AWAY_THRESHOLD, 1 / 8.0) - 1.0) 


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

	if (show_grid && screen_grid(gl_FragCoord.xy, rows))
		frag = vec4(1, 1, 1, 0.3);

	// Tile state:
	//  1  ; if alive
	//  0  ; 0 neighbours
	// -n  ; n is the number of neighbours of the PREVIOUS state
	//  n \in [-8, 1] ; normal state
	//  n \in [2, 11] ; hovered state (without pattern bit)
	//  n \in [22, 31] ; hovered state (with pattern bit)
	// Tile age:
	//  k >  0; k ticks alive
	//  k <= 0; k ticks dead

	// Mouse hovering a pattern.
	if (tile_state >= 2) {
		frag = vec4(0, 1, 0, 1) * (tile_state / 31.0);
		return;
	}

	// Alive.
	if (tile_age > 0) {
		float x = tile_age / BREATHE_PERIOD;
		float age = 0.7 + 0.3 * cos(2 * PI * x);
		frag = vec4(1, 0, 0, 1) * age;
	} // Previous neighbours.
	else if (tile_age < 0 && tile_state < 0) {
		frag = vec4(0.7, 0.2, 0.4, 1) * mix(0.2, 1.0, tile_state / -8.0f);	
	} // Dead.
	else if (tile_age >= -FADE_AWAY_PERIOD) {
		// Note: for this part to work as expected, tile age should be
		// initialized to a number less than FADE_AWAY_PERIOD ... TODO
		float x = tile_age / -FADE_AWAY_PERIOD * FADE_AWAY_THRESHOLD_X;
		frag = vec4(1, 0, 0.5, 0.6) * pow(x + 1, -8);	
	}
}