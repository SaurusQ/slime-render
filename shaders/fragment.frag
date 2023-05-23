#version 330 core

out vec4 fragColor;
uniform vec2 circleCenter;
uniform float circleRadius;

void main()
{
    vec2 pixelPos = (gl_FragCoord.xy - circleCenter) / circleRadius;
    float distance = length(pixelPos);

    if (distance <= 1.0)
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);  // Inside the circle, set red color
    else
        discard;  // Outside the circle, discard the fragment
}
