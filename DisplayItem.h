#ifndef DISPLAYITEM_H
#define DISPLAYITEM_H

#include <GL/glew.h>

#include "Common.h"
#include "ShaderProgram.h"
#include "TransformMatrix.h"

class DisplayItem
{
public:
    DisplayItem() :
        Vao_(0),
        VertexVbo_(0),
        ConnectVbo_(0),
        NormalVbo_(0),
        ColorVbo_(0),
        SingleColor_(make_float4(0.f, 0.f, 0.f, 1.f)),
        bHasSingleColor_(false),
        NumElements_(-1),
        Shader_(NULL),
        ProjMatrix_(NULL),
        VisMatrix_(NULL)
    {}

    ~DisplayItem();

    void Init(ShaderProgram* Shaders, TransformMatrix* Projection, TransformMatrix* Visualization,
              const float3* Vertices, const float3* Normals, const uint NumVertices, const uint3* Connectivity, const uint NumElements,
              const float4* Colors, const uint NumColors);
    void Draw();

private:
    GLuint Vao_;
    GLuint VertexVbo_;
    GLuint ConnectVbo_;
    GLuint NormalVbo_;
    GLuint ColorVbo_;
    float4 SingleColor_;
    bool bHasSingleColor_;

    int NumElements_;

    ShaderProgram* Shader_;
    TransformMatrix* ProjMatrix_;
    TransformMatrix* VisMatrix_;
    TransformMatrix ModelMatrix_;

};

#endif // DISPLAYITEM_H
