#include "ShaderProgram.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

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
    if ( ShaderSourceFile.fail() )
    {
       cerr << "Failed to open " << FileName << ": " << strerror(errno) << endl;
       return;
    }

    stringstream ShaderSourceStream;
    ShaderSourceStream << ShaderSourceFile.rdbuf();
    ShaderSourceFile.close();

    string ShaderSource = ShaderSourceStream.str();
    const int ShaderSize = ShaderSource.size();

    char* SourceChars = new char[ShaderSize + 1];

    if ( SourceChars != NULL )
    {
        strcpy( SourceChars, ShaderSource.c_str() );
        GLuint ShaderObject = glCreateShader( ShaderType );
        glShaderSource( ShaderObject, 1, &SourceChars, NULL );
        glCompileShader( ShaderObject );
        glAttachShader( ProgramId_, ShaderObject );

        // Get and print compile log
        int InfoLogLength = 0;
        glGetShaderiv( ShaderObject, GL_INFO_LOG_LENGTH, &InfoLogLength );
        if ( InfoLogLength > 1 )
        {
           char* InfoLog = new char[InfoLogLength + 1];
           int CharsWritten = 0;
           glGetShaderInfoLog( ShaderObject, InfoLogLength, &CharsWritten, InfoLog );
           cout << endl << InfoLog << endl;
           delete[] InfoLog;
        }

        delete [] SourceChars;
    }
}

void ShaderProgram::LinkProgram()
{
    glLinkProgram(ProgramId_);

    // afficher le message d'erreur, le cas échéant
    int InfoLogLength = 0;
    glGetProgramiv( ProgramId_, GL_INFO_LOG_LENGTH, &InfoLogLength );
    if ( InfoLogLength > 1 )
    {
       char* InfoLog = new char[InfoLogLength + 1];
       int CharsWritten = 0;
       glGetProgramInfoLog( ProgramId_, InfoLogLength, &CharsWritten, InfoLog );
       cout << "Link log: " << endl
            << InfoLog << endl;
       delete[] InfoLog;
    }

    RetrieveLocations();
}

void ShaderProgram::UseProgram()
{
    glUseProgram(ProgramId_);
}

void ShaderProgram::RetrieveLocations()
{
    ProjMatrixLocation_ = glGetUniformLocation(ProgramId_, "ProjMatrix");   CheckLocation(ProjMatrixLocation_, "ProjMatrix");
    VisMatrixLocation_ = glGetUniformLocation(ProgramId_, "VisMatrix");     CheckLocation(VisMatrixLocation_, "VisMatrix");
    ModelMatrixLocation_ = glGetUniformLocation(ProgramId_, "ModelMatrix"); CheckLocation(ModelMatrixLocation_, "ModelMatrix");

    VertexLocation_ = glGetAttribLocation(ProgramId_, "Vertex"); CheckLocation(VertexLocation_, "Vertex");
    ColorLocation_ = glGetAttribLocation(ProgramId_, "Color");   CheckLocation(ColorLocation_, "Color");
}

void ShaderProgram::CheckLocation(GLint Location, const string VarName)
{
    if (Location < 0)
    {
        cerr << "[WARNING] This program has no " << VarName << " location" << endl;
    }
}
