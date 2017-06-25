#ifndef RAYMARCHINGTEXTURE_H
#define RAYMARCHINGTEXTURE_H

#include <GL/glew.h>

#include "HostDeviceCode.h"

class RayMarchingTexture
{
public:
    RayMarchingTexture();

    void Init();
    void Update();

    GLuint GetTextureId() const { return Texture_; }

private:
    RayMarchingParam Param_;

    GLuint Texture_; //!< OpenGL texture id
    cudaGraphicsResource* TexResource_; //!< CUDA reference to the OpenGL texture
    cudaArray* TexArray_; //!< OpenGl texture data as a CUDA array
    uint TexDataSize_; //!< Size of the texture data (in bytes)

    void InitTexture();
    void MapBuffers();
    void UnmapBuffers();
};

#endif // RAYMARCHINGTEXTURE_H
