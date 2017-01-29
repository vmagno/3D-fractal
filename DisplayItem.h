#ifndef DISPLAYITEM_H
#define DISPLAYITEM_H

#include <GL/glew.h>

#include "Common.h"
#include "TransformMatrix.h"

struct SDL_Surface;
class ShaderProgram;

class DisplayItem
{
public:
    DisplayItem();

    virtual ~DisplayItem();

    virtual void Init(ShaderProgram* Shaders, TransformMatrix* Projection, TransformMatrix* Visualization,
                      const float3* Vertices, const float3* Normals, const uint NumVertices,
                      const uint3* Connectivity, const uint NumElements,
                      const float4* Colors, const uint NumColors,
                      const SDL_Surface* TexImage, const float2* TexCoord);


    void Draw();

    bool HasSingleColor() const { return bHasSingleColor_; }
    const float* GetSingleColor() const { return (float*)&SingleColor_; }

    bool HasTexture() const { return bHasTexture_; }
    GLuint GetTextureId() const { return Texture_; }
    void SetTexture(GLuint TextureId);

    const TransformMatrix& GetProjMatrix() const { return *ProjMatrix_; }
    const TransformMatrix& GetVisMatrix() const { return *VisMatrix_; }
    const TransformMatrix& GetModelMatrix() const { return ModelMatrix_; }

protected:
    const uint NumVBO_ = 4;
    int NumElements_;
    GLuint* VBOs_;
    GLuint Texture_;

protected:
    enum
    {
        VERTEX_VBO_ID,
        CONNECT_VBO_ID,
        NORMAL_VBO_ID,
        COLOR_VBO_ID
    };


private:
    GLuint Vao_;

    float4 SingleColor_;
    bool bHasSingleColor_;

    bool bHasTexture_;
    GLuint TextureCoordVBO_;

    ShaderProgram* Shader_;
    TransformMatrix* ProjMatrix_;
    TransformMatrix* VisMatrix_;
    TransformMatrix ModelMatrix_;


};

#endif // DISPLAYITEM_H
