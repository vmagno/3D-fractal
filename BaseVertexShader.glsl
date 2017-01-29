#version 410

uniform mat4 ModelMatrix;
uniform mat4 VisMatrix;
uniform mat4 ProjMatrix;

layout(location=0) in vec4 Vertex;
layout(location=1) in vec4 Color;
layout(location=2) in vec2 TexCoord;

out Attribs {
    vec4 Color;
    vec2 TexCoord;
} AttribsOut;

void main( void )
{
   gl_Position = ProjMatrix * VisMatrix * ModelMatrix * Vertex;
   AttribsOut.Color = Color;
   AttribsOut.TexCoord = TexCoord;
}
