#include "DisplayItem.h"
#include "LightShader.h"

using namespace std;


DisplayItem::DisplayItem() :
    NumElements_(-1),
    VBOs_(NULL),
    Vao_(0),
    SingleColor_(make_float4(0.f, 0.f, 0.f, 1.f)),
    bHasSingleColor_(false),
    Shader_(NULL),
    ProjMatrix_(NULL),
    VisMatrix_(NULL)
{
}

DisplayItem::~DisplayItem()
{
    if (Vao_ > 0) glDeleteVertexArrays(1, &Vao_);
    if (VBOs_ != NULL)
    {
        glDeleteBuffers(NumVBO_, VBOs_);
        delete [] VBOs_;
    }
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

    VBOs_ = new GLuint[NumVBO_];

    glGenVertexArrays( 1, &Vao_ );
    glGenBuffers(NumVBO_, VBOs_);

    glBindVertexArray(Vao_);

    // Init vertices
    glBindBuffer(GL_ARRAY_BUFFER, VBOs_[VERTEX_VBO_ID]);
    glBufferData(GL_ARRAY_BUFFER, NumVertices * sizeof(float3), Vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(Shader_->GetVertexLocation(), 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader_->GetVertexLocation());

    // Init connectivity
    NumElements_ = NumElements;
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOs_[CONNECT_VBO_ID]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, NumElements * sizeof(uint3), Connectivity, GL_STATIC_DRAW);

    // Init normals
    if (Normals != NULL)
    {
        glBindBuffer(GL_ARRAY_BUFFER, VBOs_[NORMAL_VBO_ID]);
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

        glBindBuffer(GL_ARRAY_BUFFER, VBOs_[COLOR_VBO_ID]);
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

    glBindVertexArray(Vao_);

    glDrawElements(GL_TRIANGLES, NumElements_ * 3, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}
