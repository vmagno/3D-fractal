#version 410

uniform mat4 ModelMatrix;
uniform mat4 VisMatrix;
uniform mat4 ProjMatrix;
uniform mat3 NormalMatrix;

uniform vec3 LightPosition;

layout(location=0) in vec3 Vertex;
layout(location=1) in vec3 Normal;
layout(location=2) in vec4 Color;

out Attribs
{
    vec4 Color;
    vec3 Normal;
    vec3 LightDir;
    vec3 ObsDir;
} AttribsOut;

void main( void )
{
   gl_Position = ProjMatrix * VisMatrix * ModelMatrix * vec4(Vertex, 1.f);
   AttribsOut.Normal = NormalMatrix * Normal;

   vec3 CamSpacePos = vec3(VisMatrix * ModelMatrix * vec4(Vertex, 1.f));

   AttribsOut.LightDir = vec3( (VisMatrix * vec4(LightPosition, 1)).xyz - CamSpacePos );
   AttribsOut.ObsDir = -CamSpacePos;
   AttribsOut.Color = Color;
}
