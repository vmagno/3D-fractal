#ifndef LIGHTSOURCE_H
#define LIGHTSOURCE_H

#include "Common.h"

class LightSource
{
public:
    LightSource()
        : Position_(make_float3(-0.f, 0.f, 5.f))
        , Ambient_(make_float4(1.f, 1.f, 1.f, 1.f))
        , Diffuse_(make_float4(1.f, 1.f, 1.f, 1.f))
        , Specular_(make_float4(1.f, 1.f, 1.f, 1.f))
    {
    }

    const float3& GetPosition() const { return Position_; }
    const float4& GetAmbient() const { return Ambient_; }
    const float4& GetDiffuse() const { return Diffuse_; }
    const float4& GetSpecular() const { return Specular_; }

    void SetPosition(const float3& Position) { Position_ = Position; }
    void SetAmbient(const float4& Ambient) { Ambient_ = Ambient; }
    void SetDiffuse(const float4& Diffuse) { Diffuse_ = Diffuse; }
    void SetSpecular(const float4& Specular) { Specular_ = Specular; }

    void Draw();

private:
    float3 Position_;

    float4 Ambient_;
    float4 Diffuse_;
    float4 Specular_;
};

#endif // LIGHTSOURCE_H
