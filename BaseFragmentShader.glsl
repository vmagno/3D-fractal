#version 410

uniform int bUseTexture;

uniform sampler2D TheTexture;

in Attribs {
    vec4 Color;
    vec2 TexCoord;
} AttribsIn;

out vec4 FragColor;

void main( void )
{
    vec4 TargetColor = AttribsIn.Color;
    if (bUseTexture == 1)
    {
        TargetColor = mix(TargetColor, texture(TheTexture, AttribsIn.TexCoord), 1.f);
    }
    FragColor = TargetColor;

//    if (bUseTexture == 1)
//    {
//        FragColor = vec4(clamp(AttribsIn.TexCoord, 0, 1), 0, 1);
//    }
//    if (bUseTexture == 1)
//    {
//        FragColor = texture(TheTexture, AttribsIn.TexCoord);
//        if (AttribsIn.TexCoord.x < 0.f || AttribsIn.TexCoord.y < 0.f || AttribsIn.TexCoord.x > 1.f || AttribsIn.TexCoord.y > 1.f) FragColor = AttribsIn.Color;
//    }
//    else
//        FragColor = AttribsIn.Color;
}
