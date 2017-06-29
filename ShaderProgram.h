#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include <iostream>
#include <string>

#include <GL/glew.h>

class DisplayItem;

class ShaderProgram
{
public:
    ShaderProgram()
        : ProgramId_(0)
        , VertexLocation_(-1)
        , ColorLocation_(-1)
        , TextureLocation_(-1)
        , TexCoordLocation_(-1)
        , UseTextureLocation_(-1)
        , ProjMatrixLocation_(-1)
        , VisMatrixLocation_(-1)
        , ModelMatrixLocation_(-1)
    {
    }

    void CreateProgram();
    void ReadShaderSource(std::string FileName, GLenum ShaderType);
    void         LinkProgram();
    virtual void UseProgram(const DisplayItem* Model);

    virtual void RetrieveLocations();

    //    //GLuint GetProgramId() { return ProgramId_; }
    GLint GetVertexLocation() { return VertexLocation_; }
    GLint GetColorLocation() { return ColorLocation_; }
    GLint GetTexCoordLocation() { return TexCoordLocation_; }
    GLint GetTextureLocation() { return TextureLocation_; }
    //    GLint GetProjLocation() { return ProjMatrixLocation_; }
    //    GLint GetVisLocation() { return VisMatrixLocation_; }
    //    GLint GetModelLocation() { return ModelMatrixLocation_; }

protected:
    GLuint ProgramId_;

    void CheckLocation(GLint Location, const std::string VarName);

private:
    GLint VertexLocation_;
    GLint ColorLocation_;

    GLint TextureLocation_;
    GLint TexCoordLocation_;
    GLint UseTextureLocation_;

    GLint ProjMatrixLocation_;
    GLint VisMatrixLocation_;
    GLint ModelMatrixLocation_;
};

#endif // SHADERPROGRAM_H
