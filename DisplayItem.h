#ifndef DISPLAYITEM_H
#define DISPLAYITEM_H

#include <GL/glew.h>

#include "Common.h"
#include "ShaderProgram.h"
#include "TransformMatrix.h"

class DisplayItem
{
public:
    DisplayItem();

    virtual ~DisplayItem();

    virtual void Init(ShaderProgram* Shaders, TransformMatrix* Projection, TransformMatrix* Visualization,
                      const float3* Vertices, const float3* Normals, const uint NumVertices,
                      const uint3* Connectivity, const uint NumElements,
                      const float4* Colors, const uint NumColors);
    void Draw();

protected:
    const uint NumVBO_ = 4;
    int NumElements_;
    GLuint* VBOs_;

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


    ShaderProgram* Shader_;
    TransformMatrix* ProjMatrix_;
    TransformMatrix* VisMatrix_;
    TransformMatrix ModelMatrix_;


};

#endif // DISPLAYITEM_H
