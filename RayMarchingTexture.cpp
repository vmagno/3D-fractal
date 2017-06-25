#include "RayMarchingTexture.h"

#include <cmath>

#include <cuda_gl_interop.h>

#include "Kernels.h"

const uint3 DEFAULT_BLOCK_SIZE = make_uint3(256, 1, 1);
const uint3 DEFAULT_NUM_BLOCKS = make_uint3(0, 0, 0);

const uint2 DEFAULT_SIZE = make_uint2(256, 256);
const float3 DEFAULT_CAMERA_POSITION = make_float3(0.f, 0.f, -1.f);
const float3 DEFAULT_CAMERA_DIRECTION = make_float3(0.f, 0.f, 1.f);
const float3 DEFAULT_CAMERA_UP = make_float3(0.f, 1.f, 0.f);

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
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
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
    LaunchRayMarching(Param_);
    CudaCheck(cudaDeviceSynchronize());
    MapBuffers();
    CudaCheck(cudaMemcpyToArray(TexArray_, 0, 0, Param_.TexCuda, TexDataSize_, cudaMemcpyDeviceToDevice));
    UnmapBuffers();
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
