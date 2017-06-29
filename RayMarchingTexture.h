#ifndef RAYMARCHINGTEXTURE_H
#define RAYMARCHINGTEXTURE_H

#include <GL/glew.h>

#include "BasicTimer.h"
#include "HostDeviceCode.h"

class RayMarchingTexture
{
public:
    RayMarchingTexture();

    void Init();
    void Update();

    GLuint GetTextureId() const { return Texture_; }
    void SetCameraInfo(const float3 Position, const float3 Direction, const float3 Up);
    float GetDistanceFromCamera();
    void SetPerspective(float FOVy, float AspectRatio, float zNear, float zFar);

private:
    RayMarchingParam Param_;

    GLuint                Texture_;     //!< OpenGL texture id
    cudaGraphicsResource* TexResource_; //!< CUDA reference to the OpenGL texture
    cudaArray*            TexArray_;    //!< OpenGl texture data as a CUDA array
    uint                  TexDataSize_; //!< Size of the texture data (in bytes)

    void InitTexture();
    void MapBuffers();
    void UnmapBuffers();

    BasicTimer MarchTimer_;
};

#endif // RAYMARCHINGTEXTURE_H
