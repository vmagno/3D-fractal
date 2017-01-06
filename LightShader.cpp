#include "LightShader.h"

#include <iostream>

#include "LightSource.h"
#include "Material.h"

using namespace std;

void LightShader::RetrieveLocations()
{
    ShaderProgram::RetrieveLocations();

    NormalLocation_ = glGetAttribLocation(ProgramId_, "Normal"); CheckLocation(NormalLocation_, "Normal");
    NormalMatrixLocation_ = glGetUniformLocation(ProgramId_, "NormalMatrix"); CheckLocation(NormalMatrixLocation_, "NormalMatrix");

    ModelAmbientLocation_ = glGetUniformLocation(ProgramId_, "LightModelAmbient"); CheckLocation(ModelAmbientLocation_, "LightModelAmbient");
    LightPosLocation_ = glGetUniformLocation(ProgramId_, "LightPosition"); CheckLocation(LightPosLocation_, "LightPosition");
    LightAmbientLocation_ = glGetUniformLocation(ProgramId_, "LightAmbient"); CheckLocation(LightAmbientLocation_, "LightAmbient");
    LightDiffuseLocation_ = glGetUniformLocation(ProgramId_, "LightDiffuse"); CheckLocation(LightDiffuseLocation_, "LightDiffuse");
    LightSpecularLocation_ = glGetUniformLocation(ProgramId_, "LightSpecular"); CheckLocation(LightSpecularLocation_, "LightSpecular");

    MatEmissiveLocation_ = glGetUniformLocation(ProgramId_, "MatEmissive"); CheckLocation(MatEmissiveLocation_, "MatEmissive");
    MatAmbientLocation_ = glGetUniformLocation(ProgramId_, "MatAmbient"); CheckLocation(MatAmbientLocation_, "MatAmbient");
    MatDiffuseLocation_ = glGetUniformLocation(ProgramId_, "MatDiffuse"); CheckLocation(MatDiffuseLocation_, "MatDiffuse");
    MatSpecularLocation_ = glGetUniformLocation(ProgramId_, "MatSpecular"); CheckLocation(MatSpecularLocation_, "MatSpecular");
    MatShininessLocation_ = glGetUniformLocation(ProgramId_, "MatShininess"); CheckLocation(MatShininessLocation_, "MatShininess");
}

void LightShader::UseProgram()
{
    ShaderProgram::UseProgram();

    glUniform3fv(LightPosLocation_, 1, (float*)&(Light_->GetPosition()));
    glUniform4fv(LightAmbientLocation_, 1, (float*)&(Light_->GetAmbient()));
    glUniform4fv(LightDiffuseLocation_, 1, (float*)&(Light_->GetDiffuse()));
    glUniform4fv(LightSpecularLocation_, 1, (float*)&(Light_->GetSpecular()));

    glUniform4fv(MatEmissiveLocation_, 1, (float*)&(Material_->GetEmissive()));
    glUniform4fv(MatAmbientLocation_, 1, (float*)&(Material_->GetAmbient()));
    glUniform4fv(MatDiffuseLocation_, 1, (float*)&(Material_->GetDiffuse()));
    glUniform4fv(MatSpecularLocation_, 1, (float*)&(Material_->GetSpecular()));
    glUniform1f(MatShininessLocation_, Material_->GetShininess());
}
