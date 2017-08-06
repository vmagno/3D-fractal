#include "DisplayItem.h"

#include <SDL2/SDL.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "Common.h"
#include "HostDeviceCode.h"
#include "LightShader.h"

using namespace std;

DisplayItem::DisplayItem()
    : NumElements_(-1)
    , VBOs_(NULL)
    , Texture_(0)
    , Vao_(0)
    , SingleColor_(make_float4(0.f, 0.f, 0.f, 1.f))
    , bHasSingleColor_(false)
    , bHasTexture_(false)
    , TextureCoordVBO_(0)
    , Shader_(NULL)
    , ProjMatrix_(NULL)
    , VisMatrix_(NULL)
    , ModelMatrix_("ModelMatrix")
{
}

DisplayItem::~DisplayItem()
{
    if (Vao_ > 0) glDeleteVertexArrays(1, &Vao_);
    if (VBOs_ != NULL)
    {
        glDeleteBuffers(NumVBO_, VBOs_);
        delete[] VBOs_;
    }
}

void DisplayItem::Init(ShaderProgram* Shaders, TransformMatrix* Projection, TransformMatrix* Visualization,
                       const float3* Vertices, const float3* Normals, const uint NumVertices, const uint3* Connectivity,
                       const uint NumElements, const float4* Colors, const uint NumColors, const SDL_Surface* TexImage,
                       const float2* TexCoord)
{
    Shader_     = Shaders;
    ProjMatrix_ = Projection;
    VisMatrix_  = Visualization;

    if (NumVertices < 1)
    {
        cerr << "[ERROR] No vertices specified" << endl;
        return;
    }

    VBOs_ = new GLuint[NumVBO_];

    glGenVertexArrays(1, &Vao_);
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
    if (dynamic_cast<LightShader*>(Shader_) != nullptr)
    {
        const LightShader* const L = dynamic_cast<LightShader*>(Shader_);
        glBindBuffer(GL_ARRAY_BUFFER, VBOs_[NORMAL_VBO_ID]);
        glBufferData(GL_ARRAY_BUFFER, NumVertices * sizeof(float3), Normals, GL_STATIC_DRAW);
        glVertexAttribPointer(L->GetNormalLocation(), 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(L->GetNormalLocation());
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
        SingleColor_     = Colors[0];
    }

    if (TexImage != NULL)
    {
        glActiveTexture(GL_TEXTURE0);
        bHasTexture_ = true;
        glGenTextures(1, &Texture_);
        glBindTexture(GL_TEXTURE_2D, Texture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, TexImage->w, TexImage->h, 0, GL_BGR, GL_UNSIGNED_BYTE,
                     TexImage->pixels);
    }

    glGenBuffers(1, &TextureCoordVBO_);
    glBindBuffer(GL_ARRAY_BUFFER, TextureCoordVBO_);
    glBufferData(GL_ARRAY_BUFFER, NumVertices * sizeof(float2), TexCoord, GL_STATIC_DRAW);
    glVertexAttribPointer(Shader_->GetTexCoordLocation(), 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader_->GetTexCoordLocation());

    glGetError();

    glBindVertexArray(0);
}

void DisplayItem::Draw()
{
    Shader_->UseProgram(this);

    if (bHasTexture_)
    {
        glEnable(GL_TEXTURE_2D);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, Texture_);
        glUniform1i(Shader_->GetTextureLocation(), 0);
    }

    glBindVertexArray(Vao_);

    glDrawElements(GL_TRIANGLES, NumElements_ * 3, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void DisplayItem::SetTexture(GLuint TextureId)
{
    if (TextureId > 0)
    {
        Texture_     = TextureId;
        bHasTexture_ = true;
    }
}
