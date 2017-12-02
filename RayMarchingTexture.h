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

    inline GLuint GetTextureId() const { return Texture_; }
    void SetCameraInfo(const float3 Position, const float3 Direction, const float3 Up);
    float GetDistanceFromCamera() const;
    void SetPerspective(float FOVy, float AspectRatio, float zNear, float zFar);

    inline void ResetView()
    {
        NextStep_             = RayMarchingStep::HalfRes;
        Param_.CurrentSubstep = 0;
    }

    void IncreaseMaxSteps();
    void DecreaseMaxSteps();
    void IncreaseMinDist();
    void DecreaseMinDist();

    void IncreaseEpsilon() { Param_.EpsilonFactor *= 1.110f; }
    void DecreaseEpsilon() { Param_.EpsilonFactor *= 0.911f; }

    void PrintMarchingParam();

private:
    RayMarchingParam Param_;
    RayMarchingStep  NextStep_;

    GLuint                Texture_;     //!< OpenGL texture id
    cudaGraphicsResource* TexResource_; //!< CUDA reference to the OpenGL texture
    cudaArray*            TexArray_;    //!< OpenGl texture data as a CUDA array
    uint                  TexDataSize_; //!< Size of the texture data (in bytes)

    void InitTexture();
    void MapBuffers();
    void UnmapBuffers();

    BasicTimer MarchTimer_;
    BasicTimer CopyTimer_;
};

#endif // RAYMARCHINGTEXTURE_H
