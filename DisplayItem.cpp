#include "DisplayItem.h"
#include "LightShader.h"

using namespace std;

DisplayItem::~DisplayItem()
{
    if (Vao_ > 0) glDeleteVertexArrays(1, &Vao_);
    if (VertexVbo_ > 0) glDeleteBuffers(1, &VertexVbo_);
    if (ConnectVbo_ > 0) glDeleteBuffers(1, &ConnectVbo_);
    if (NormalVbo_ > 0) glDeleteBuffers(1, &NormalVbo_);
    if (ColorVbo_ > 0) glDeleteBuffers(1, &ColorVbo_);
}

void DisplayItem::Init(ShaderProgram* Shaders, TransformMatrix* Projection, TransformMatrix* Visualization, const float3* Vertices, const float3* Normals, const uint NumVertices,
                       const uint3* Connectivity, const uint NumElements, const float4* Colors, const uint NumColors)
{
    Shader_ = Shaders;
    ProjMatrix_ = Projection;
    VisMatrix_ = Visualization;

    if (NumVertices < 1)
    {
        cerr << "[ERROR] No vertices specified" << endl;
        return;
    }

    glGenVertexArrays( 1, &Vao_ );
//    cout << "Vao = " << Vao_ << endl;

    glBindVertexArray(Vao_);

    // Init vertices
    glGenBuffers(1, &VertexVbo_);
//    cout << "vertex vbo = " << VertexVbo_ << endl;
    glBindBuffer(GL_ARRAY_BUFFER, VertexVbo_);
    glBufferData(GL_ARRAY_BUFFER, NumVertices * sizeof(float3), Vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(Shader_->GetVertexLocation(), 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader_->GetVertexLocation());

    // Init connectivity
    NumElements_ = NumElements;
    glGenBuffers(1, &ConnectVbo_);
//    cout << "ConnectVbo_ vbo = " << ConnectVbo_ << endl;
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ConnectVbo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, NumElements * sizeof(uint3), Connectivity, GL_STATIC_DRAW);

    // Init normals
    if (Normals != NULL)
    {
        glGenBuffers(1, &NormalVbo_);
        glBindBuffer(GL_ARRAY_BUFFER, NormalVbo_);
//        cout << "NormalVbo_ = " << NormalVbo_ << endl;
        glBufferData(GL_ARRAY_BUFFER, NumVertices * sizeof(float3), Normals, GL_STATIC_DRAW);
        glVertexAttribPointer(((LightShader*)Shader_)->GetNormalLocation(), 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(((LightShader*)Shader_)->GetNormalLocation());
    }


    // Init color(s)
    if (NumColors > 1)
    {
        if (NumColors != NumVertices)
        {
            cerr << "[WARNING] Not all vertices have a color" << endl;
        }

        glGenBuffers(1, &ColorVbo_);
        glBindBuffer(GL_ARRAY_BUFFER, ColorVbo_);
//        cout << "ColorVbo_ = " << ColorVbo_ << endl;
        glBufferData(GL_ARRAY_BUFFER, NumColors * sizeof(float4), Colors, GL_STATIC_DRAW);
        glVertexAttribPointer(Shader_->GetColorLocation(), 4, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader_->GetColorLocation());
    }
    else
    {
        bHasSingleColor_ = true;
        SingleColor_ = Colors[0];
    }

    glBindVertexArray(0);

}

void DisplayItem::Draw()
{
    Shader_->UseProgram();

    if (bHasSingleColor_)
    {
        glVertexAttrib4fv(Shader_->GetColorLocation(), (float*)&SingleColor_);
    }

    glUniformMatrix4fv(Shader_->GetProjLocation(), 1, GL_FALSE, *ProjMatrix_);
    glUniformMatrix4fv(Shader_->GetVisLocation(), 1, GL_FALSE, *VisMatrix_);
    glUniformMatrix4fv(Shader_->GetModelLocation(), 1, GL_FALSE, ModelMatrix_);

    glUniformMatrix3fv(((LightShader*)Shader_)->GetNormalMatrixLocation(), 1, GL_FALSE,
                       glm::value_ptr( glm::inverse( glm::mat3(VisMatrix_->GetMatrix() * ModelMatrix_.GetMatrix()) ) ) );

//    cout << "Proj matrix: " << *ProjMatrix_ << endl;
//    cout << "Vis matrix: " << *VisMatrix_ << endl;
//    cout << "Model matrix: " << ModelMatrix_ << endl;

    glBindVertexArray(Vao_);

    glDrawElements(GL_TRIANGLES, NumElements_ * 3, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}
