#include "ShaderProgram.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "DisplayItem.h"

using namespace std;

void ShaderProgram::CreateProgram()
{
    cout << "Creating a shader program" << endl;
    ProgramId_ = glCreateProgram();
}

void ShaderProgram::ReadShaderSource(string FileName, GLenum ShaderType)
{
    if (ProgramId_ == 0) CreateProgram();

    ifstream ShaderSourceFile(FileName);
    if (ShaderSourceFile.fail())
    {
        cerr << "Failed to open " << FileName << ": " << strerror(errno) << endl;
        return;
    }

    stringstream ShaderSourceStream;
    ShaderSourceStream << ShaderSourceFile.rdbuf();
    ShaderSourceFile.close();

    string    ShaderSource = ShaderSourceStream.str();
    const int ShaderSize   = ShaderSource.size();

    GLchar* SourceChars = new GLchar[ShaderSize + 1];

    if (SourceChars != NULL)
    {
        strcpy(SourceChars, ShaderSource.c_str());
        GLuint ShaderObject = glCreateShader(ShaderType);
        glShaderSource(ShaderObject, 1, (const char**)&SourceChars, NULL);
        glCompileShader(ShaderObject);
        glAttachShader(ProgramId_, ShaderObject);

        // Get and print compile log
        int InfoLogLength = 0;
        glGetShaderiv(ShaderObject, GL_INFO_LOG_LENGTH, &InfoLogLength);
        if (InfoLogLength > 1)
        {
            char* InfoLog      = new char[InfoLogLength + 1];
            int   CharsWritten = 0;
            glGetShaderInfoLog(ShaderObject, InfoLogLength, &CharsWritten, InfoLog);
            cout << endl << InfoLog << endl;
            delete[] InfoLog;
        }

        delete[] SourceChars;
    }
}

void ShaderProgram::LinkProgram()
{
    glLinkProgram(ProgramId_);

    // afficher le message d'erreur, le cas échéant
    int InfoLogLength = 0;
    glGetProgramiv(ProgramId_, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if (InfoLogLength > 1)
    {
        char* InfoLog      = new char[InfoLogLength + 1];
        int   CharsWritten = 0;
        glGetProgramInfoLog(ProgramId_, InfoLogLength, &CharsWritten, InfoLog);
        cout << "Link log: " << endl << InfoLog << endl;
        delete[] InfoLog;
    }

    RetrieveLocations();
}

void ShaderProgram::UseProgram(const DisplayItem* Model)
{
    glUseProgram(ProgramId_);

    glUniformMatrix4fv(ProjMatrixLocation_, 1, GL_FALSE, Model->GetProjMatrix());
    glUniformMatrix4fv(VisMatrixLocation_, 1, GL_FALSE, Model->GetVisMatrix());
    glUniformMatrix4fv(ModelMatrixLocation_, 1, GL_FALSE, Model->GetModelMatrix());

    if (Model->HasSingleColor())
    {
        glVertexAttrib4fv(ColorLocation_, Model->GetSingleColor());
    }

    if (Model->HasTexture())
    {
        glUniform1i(UseTextureLocation_, 1);
    }
    else
    {
        glUniform1i(UseTextureLocation_, 0);
    }
}

void ShaderProgram::RetrieveLocations()
{
    ProjMatrixLocation_ = glGetUniformLocation(ProgramId_, "ProjMatrix");
    CheckLocation(ProjMatrixLocation_, "ProjMatrix");
    VisMatrixLocation_ = glGetUniformLocation(ProgramId_, "VisMatrix");
    CheckLocation(VisMatrixLocation_, "VisMatrix");
    ModelMatrixLocation_ = glGetUniformLocation(ProgramId_, "ModelMatrix");
    CheckLocation(ModelMatrixLocation_, "ModelMatrix");

    VertexLocation_ = glGetAttribLocation(ProgramId_, "Vertex");
    CheckLocation(VertexLocation_, "Vertex");
    ColorLocation_ = glGetAttribLocation(ProgramId_, "Color");
    CheckLocation(ColorLocation_, "Color");

    TextureLocation_ = glGetUniformLocation(ProgramId_, "TheTexture");
    CheckLocation(TextureLocation_, "TheTexture");
    TexCoordLocation_ = glGetAttribLocation(ProgramId_, "TexCoord");
    CheckLocation(TexCoordLocation_, "TexCoord");
    UseTextureLocation_ = glGetUniformLocation(ProgramId_, "bUseTexture");
    CheckLocation(UseTextureLocation_, "bUseTexture");
}

void ShaderProgram::CheckLocation(GLint Location, const string VarName)
{
    if (Location < 0)
    {
        cerr << "[WARNING] This program has no " << VarName << " location" << endl;
    }
}
