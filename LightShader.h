#ifndef LIGHTSHADER_H
#define LIGHTSHADER_H

#include <GL/glew.h>

#include "ShaderProgram.h"

class LightSource;
class Material;

class LightShader : public ShaderProgram
{
public:
    LightShader()
        : NormalLocation_(-1)
    {
    }

    void RetrieveLocations() override;
    void UseProgram(const DisplayItem* Model) override;

    void SetLightSource(LightSource* Light) { Light_ = Light; }
    void SetMaterial(Material* Material) { Material_ = Material; }

    GLint GetNormalLocation() const { return NormalLocation_; }
    //    GLint GetNormalMatrixLocation() { return NormalMatrixLocation_; }

private:
    GLint NormalLocation_;
    GLint NormalMatrixLocation_;

    GLint ModelAmbientLocation_;

    GLint LightPosLocation_;
    GLint LightAmbientLocation_;
    GLint LightDiffuseLocation_;
    GLint LightSpecularLocation_;

    GLint MatEmissiveLocation_;
    GLint MatAmbientLocation_;
    GLint MatDiffuseLocation_;
    GLint MatSpecularLocation_;
    GLint MatShininessLocation_;

    LightSource* Light_;
    Material*    Material_;
};

#endif // LIGHTSHADER_H
