#version 410

uniform mat4 ModelMatrix;
uniform mat4 VisMatrix;
uniform mat4 ProjMatrix;
uniform mat3 NormalMatrix;

uniform vec3 LightPosition;

layout(location=0) in vec4 Vertex;
layout(location=1) in vec3 Normal;

out Attribs
{
    vec4 Color;
    vec3 Normal;
    vec3 LightDir;
    vec3 ObsDir;
} AttribsOut;

void main( void )
{
   gl_Position = ProjMatrix * VisMatrix * ModelMatrix * Vertex;
   AttribsOut.Normal = NormalMatrix * Normal;
   //AttribsOut.Normal = vec3(0.f, 0.f, 1.f);

   vec3 CamSpacePos = vec3(VisMatrix * ModelMatrix * Vertex);

   AttribsOut.LightDir = vec3( (VisMatrix * vec4(LightPosition, 1)).xyz - CamSpacePos );
   AttribsOut.ObsDir = -CamSpacePos;
}
