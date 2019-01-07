#version 330 core

in float enabled;
in vec2 grid_pos;

out vec4 frag;

bool grid(float aspect_ratio, vec2 screen_pos, vec2 delta, int parts, int width)
{
	return mod(screen_pos.x + delta.x, 800 / parts) < width || 
		   mod(screen_pos.y + delta.y, 600 / parts * aspect_ratio) < width;
}

void main()
{
/*
	float aspect_ratio = 800.0 / 600.0;
	vec4 pos = gl_FragCoord;
	vec2 delta = vec2(0,0);
	if (grid(aspect_ratio, pos.xy, delta, 25, 2))
		frag = vec4(0, 1, 0, 1);
	else if (grid(aspect_ratio, pos.xy, delta, 100, 1))
		frag = vec4(1, 0, 0, 1);
*/

	if (enabled > 0.5)
		frag = vec4(1, 0, 0, 1);
}