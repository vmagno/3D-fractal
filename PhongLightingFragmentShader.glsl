#version 410

uniform vec4 LightModelAmbient;

uniform vec4 LightAmbient;
uniform vec4 LightDiffuse;
uniform vec4 LightSpecular;

uniform vec4 MatEmissive;
uniform vec4 MatAmbient;
uniform vec4 MatDiffuse;
uniform vec4 MatSpecular;
uniform float MatShininess;

in Attribs
{
    vec4 Color;
    vec3 Normal;
    vec3 LightDir;
    vec3 ObsDir;
} AttribsIn;

out vec4 FragColor;


void main( void )
{
   vec3 L = normalize(AttribsIn.LightDir);
   vec3 N = normalize(AttribsIn.Normal);
   vec3 O = normalize(AttribsIn.ObsDir);

   vec4 Color = MatEmissive;
   Color += MatAmbient * LightAmbient;

   float NdotL = max(0.f, dot(N, L));
   if (NdotL > 0.f)
   {
       Color += MatDiffuse * LightDiffuse * NdotL;

       float NdotR = max( 0.f, dot(reflect(-L, N), O) );
       Color += MatSpecular * LightSpecular * pow(NdotR, MatShininess);
   }

   FragColor = clamp(Color, 0.0, 1.0);
}
