#include "RayMarchingTexture.h"

#include <cmath>
#include <iomanip>
#include <iostream>

#include <cuda_gl_interop.h>

#include "Kernels.h"

using namespace std;

const uint3 DEFAULT_BLOCK_SIZE = make_uint3(256, 1, 1);
const uint3 DEFAULT_NUM_BLOCKS = make_uint3(0, 0, 0);

const uint2 DEFAULT_SIZE = make_uint2(896, 896);
const float3 DEFAULT_CAMERA_POSITION = make_float3(0.f, 0.f, -1.f);
const float3 DEFAULT_CAMERA_DIRECTION = make_float3(0.f, 0.f, 1.f);
const float3 DEFAULT_CAMERA_UP = make_float3(0.f, 1.f, 0.f);

const float DEFAULT_DEPTH = 300.f;
const float DEFAULT_WIDTH = 300.f;
const float DEFAULT_HEIGHT = 300.f;

RayMarchingTexture::RayMarchingTexture()
    : Texture_(0)
    , TexResource_(nullptr)
    , TexArray_(nullptr)
    , TexDataSize_(0)
{
    Param_.BlockSize = DEFAULT_BLOCK_SIZE;
    Param_.NumBlocks = DEFAULT_NUM_BLOCKS;
    Param_.Size = DEFAULT_SIZE;
    Param_.TexCuda = nullptr;

    Param_.CameraPos = DEFAULT_CAMERA_POSITION;
    Param_.CameraDir = DEFAULT_CAMERA_DIRECTION;
    Param_.CameraUp = DEFAULT_CAMERA_UP;

    Param_.Depth = DEFAULT_DEPTH;
    Param_.Width = DEFAULT_WIDTH;
    Param_.Height = DEFAULT_HEIGHT;
}

void RayMarchingTexture::Init()
{
    const uint BlockSize = Param_.BlockSize.x * Param_.BlockSize.y * Param_.BlockSize.z;
    const uint TotalThreads = Param_.Size.x * Param_.Size.y;
    const uint NumBlocks = static_cast<uint>(ceilf(static_cast<float>(TotalThreads) / BlockSize));

    Param_.NumBlocks = make_uint3(NumBlocks, 1, 1);
    InitTexture();
}

void RayMarchingTexture::InitTexture()
{
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &Texture_);
    glBindTexture(GL_TEXTURE_2D, Texture_);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (int)Param_.Size.x, (int)Param_.Size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    TexDataSize_ = Param_.Size.x * Param_.Size.y * 4 * sizeof(GLubyte);
    CudaCheck(cudaMalloc((void**)&Param_.TexCuda, TexDataSize_));
    CudaCheck(cudaMemset(Param_.TexCuda, 0, TexDataSize_));
    CudaCheck(cudaGraphicsGLRegisterImage(&TexResource_, Texture_, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
}

void RayMarchingTexture::Update()
{
    MarchTimer_.Start();
    LaunchRayMarching(Param_);
    CudaCheck(cudaDeviceSynchronize());
    MarchTimer_.Stop();
    MapBuffers();
    CudaCheck(cudaMemcpyToArray(TexArray_, 0, 0, Param_.TexCuda, TexDataSize_, cudaMemcpyDeviceToDevice));
    UnmapBuffers();

    {
        if (MarchTimer_.GetCount() >= 60)
        {
            cout << setprecision(3) << "Average time: " << MarchTimer_.GetAverageTimeMs() << " ms" << endl;
            MarchTimer_.Reset();
        }
    }
}

void RayMarchingTexture::SetCameraInfo(const float3 Position, const float3 Direction, const float3 Up)
{
    Param_.CameraPos = Position;
    Param_.CameraDir = Direction;
    Param_.CameraUp = Up;
}

float RayMarchingTexture::GetDistanceFromCamera()
{
    return GetDistanceFromPos(Param_.CameraPos);
}

void RayMarchingTexture::SetPerspective(float FOVy, float AspectRatio, float /*zNear*/, float zFar)
{
   Param_.Depth = zFar;
   Param_.Height = tanf(FOVy * 3.14159265f / 180.f / 2.f) * Param_.Depth * 2.f;
   Param_.Width = Param_.Height * AspectRatio;
}

void RayMarchingTexture::MapBuffers()
{
    if (TexResource_)
    {
        CudaCheck(cudaGraphicsMapResources(1, &TexResource_));
        CudaCheck(cudaGraphicsSubResourceGetMappedArray(&TexArray_, TexResource_, 0, 0));
    }
}

void RayMarchingTexture::UnmapBuffers()
{
    if (TexResource_)
    {
        CudaCheck(cudaGraphicsUnmapResources(1, &TexResource_));
    }
}
