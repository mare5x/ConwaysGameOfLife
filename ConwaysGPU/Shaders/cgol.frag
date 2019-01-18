#version 330 core

uniform int rows;
uniform int cols;

in float enabled;
in vec2 grid_pos;

out vec4 frag;

bool screen_grid(float aspect_ratio, vec2 screen_pos, vec2 delta, int parts, int width)
{
	return mod(screen_pos.x + delta.x, 800.0 / parts) < float(width) || 
		   mod(screen_pos.y + delta.y, 600.0 / parts * aspect_ratio) < float(width);
}

bool world_grid(float aspect_ratio, vec2 world_pos, int parts)
{
	return mod(world_pos.x, 1.0 / parts) < 0.001 ||
		   mod(world_pos.y, 1.0 / parts * aspect_ratio) < 0.002;
}

void main()
{
	// ALWAYS WATCH OUT FOR TYPE CONVERSION WHEN WORKING WITH FLOATS AND INTS!!!

	float aspect_ratio = 800.0 / 600.0;
	// NOTE:
	// By default, gl_FragCoord assumes a lower-left origin for window 
	// coordinates and assumes pixel centers are located at half-pixel centers.
	vec4 pos = gl_FragCoord;
	vec2 delta = vec2(0,0);
	if (screen_grid(aspect_ratio, pos.xy, delta, rows, 1))
		frag = vec4(1, 1, 1, 1);

//	if (world_grid(1, grid_pos, rows))
//		frag = vec4(1, 1, 1, 1);

	if (enabled > 0.5)
		frag = vec4(1, 0, 0, 1);
}