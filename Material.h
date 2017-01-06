#ifndef MATERIAL_H
#define MATERIAL_H

#include "Common.h"

class Material
{
public:
    Material()
        :
          Emissive_(make_float4(0.f, 0.f, 0.f, 1.f)),
          Ambient_(make_float4(0.1f, 0.1f, 0.1f, 1.f)),
          Diffuse_(make_float4(0.1f, 0.1f, 1.f, 1.f)),
          Specular_(make_float4(1.f, 1.f, 1.f, 1.f)),
          Shininess_(100.f)
    {}

    float4& GetEmissive() { return Emissive_; }
    float4& GetAmbient() { return Ambient_; }
    float4& GetDiffuse() { return Diffuse_; }
    float4& GetSpecular() { return Specular_; }
    float GetShininess() { return Shininess_; }

    void SetEmissive(const float4& Emissive) { Emissive_ = Emissive; }
    void SetAmbient(const float4& Ambient) { Ambient_ = Ambient; }
    void SetDiffuse(const float4& Diffuse) { Diffuse_ = Diffuse; }
    void SetSpecular(const float4& Specular) { Specular_ = Specular; }
    void SetShininess(float Shininess) { Shininess_ = Shininess; }

private:
    float4 Emissive_;
    float4 Ambient_;
    float4 Diffuse_;
    float4 Specular_;
    float Shininess_;
};

#endif // MATERIAL_H
