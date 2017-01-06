#version 410

uniform mat4 ModelMatrix;
uniform mat4 VisMatrix;
uniform mat4 ProjMatrix;

layout(location=0) in vec4 Vertex;
layout(location=1) in vec4 Color;

out vec4 OutColor;

void main( void )
{
   gl_Position = ProjMatrix * VisMatrix * ModelMatrix * Vertex;
   OutColor = Color;
}
