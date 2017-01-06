#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include <iostream>
#include <string>

#include <GL/glew.h>

class ShaderProgram
{
public:
    ShaderProgram() :
        ProgramId_(0),
        VertexLocation_(-1),
        ColorLocation_(-1),
        ProjMatrixLocation_(-1),
        VisMatrixLocation_(-1),
        ModelMatrixLocation_(-1)
    {}

    void CreateProgram();
    void ReadShaderSource(std::string FileName, GLenum ShaderType);
    void LinkProgram();
    virtual void UseProgram();

    virtual void RetrieveLocations();

    //GLuint GetProgramId() { return ProgramId_; }
    GLint GetVertexLocation() { return VertexLocation_; }
    GLint GetColorLocation() { return ColorLocation_; }
    GLint GetProjLocation() { return ProjMatrixLocation_; }
    GLint GetVisLocation() { return VisMatrixLocation_; }
    GLint GetModelLocation() { return ModelMatrixLocation_; }

protected:
    GLuint ProgramId_;

    void CheckLocation(GLint Location, const std::string VarName);

private:
    GLint VertexLocation_;
    GLint ColorLocation_;

    GLint ProjMatrixLocation_;
    GLint VisMatrixLocation_;
    GLint ModelMatrixLocation_;
};

#endif // SHADERPROGRAM_H
