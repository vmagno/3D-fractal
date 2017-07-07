#include "RayMarchingTexture.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include <cuda_gl_interop.h>

#include "CudaMath.h"
#include "Kernels.h"

using namespace std;

const uint3 DEFAULT_BLOCK_SIZE = make_uint3(256, 1, 1);
const uint3 DEFAULT_NUM_BLOCKS = make_uint3(0, 0, 0);

// const uint2 DEFAULT_SIZE = make_uint2(240, 135);
// const uint2 DEFAULT_SIZE = make_uint2(960, 540);
const uint2  DEFAULT_SIZE             = make_uint2(1920, 1080);
const float3 DEFAULT_CAMERA_POSITION  = make_float3(0.f, 0.f, -1.f);
const float3 DEFAULT_CAMERA_DIRECTION = make_float3(0.f, 0.f, 1.f);
const float3 DEFAULT_CAMERA_UP        = make_float3(0.f, 1.f, 0.f);

const float DEFAULT_DEPTH  = 300.f;
const float DEFAULT_WIDTH  = 300.f;
const float DEFAULT_HEIGHT = 300.f;

const float DEFAULT_DISTANCE_RATIO = 0.0012f;
const uint  DEFAULT_MAX_STEPS      = 25;

const float INC_FACTOR = 1.1f;

using RMS = RayMarchingStep;

RayMarchingTexture::RayMarchingTexture()
    : NextStep_(RMS::None)
    , Texture_(0)
    , TexResource_(nullptr)
    , TexArray_(nullptr)
    , TexDataSize_(0)
{
    Param_.BlockSize = DEFAULT_BLOCK_SIZE;
    Param_.NumBlocks = DEFAULT_NUM_BLOCKS;
    Param_.Size      = DEFAULT_SIZE;
    Param_.TexCuda   = nullptr;
    Param_.Distances = nullptr;

    // Make sure the texture size uses even numbers!!!
    if (Param_.Size.x % 2 != 0) Param_.Size.x++;
    if (Param_.Size.y % 2 != 0) Param_.Size.y++;

    Param_.TotalPixels = Param_.Size.x * Param_.Size.y;

    Param_.CameraPos    = DEFAULT_CAMERA_POSITION;
    Param_.CameraDir    = DEFAULT_CAMERA_DIRECTION;
    Param_.CameraUp     = DEFAULT_CAMERA_UP;
    Param_.CameraLeft   = Cross(Param_.CameraUp, Param_.CameraDir);
    Param_.CameraRealUp = Cross(Param_.CameraDir, Param_.CameraLeft);

    Param_.Depth  = DEFAULT_DEPTH;
    Param_.Width  = DEFAULT_WIDTH;
    Param_.Height = DEFAULT_HEIGHT;

    Param_.DistanceRatio = DEFAULT_DISTANCE_RATIO;
    Param_.MinDistance   = Param_.DistanceRatio;
    Param_.MaxSteps      = DEFAULT_MAX_STEPS;

    Param_.CurrentSubstep = 0;
}

void RayMarchingTexture::Init()
{
    const uint BlockSize    = Param_.BlockSize.x * Param_.BlockSize.y * Param_.BlockSize.z;
    const uint TotalThreads = Param_.TotalPixels;
    const uint NumBlocks    = static_cast<uint>(ceilf(static_cast<float>(TotalThreads) / BlockSize));

    Param_.NumBlocks = make_uint3(NumBlocks, 1, 1);
    InitTexture();

    Param_.Print();
}

void RayMarchingTexture::InitTexture()
{
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &Texture_);
    glBindTexture(GL_TEXTURE_2D, Texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (int)Param_.Size.x, (int)Param_.Size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    TexDataSize_ = Param_.TotalPixels * 4 * sizeof(GLubyte);
    CudaCheck(cudaMalloc((void**)&Param_.TexCuda, TexDataSize_));

    CudaCheck(cudaMemset(Param_.TexCuda, 0, TexDataSize_));
    CudaCheck(cudaGraphicsGLRegisterImage(&TexResource_, Texture_, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

    CudaCheck(cudaMalloc((void**)&Param_.Distances, Param_.TotalPixels * sizeof(float)));
    CudaCheck(cudaMemset(Param_.Distances, 0, Param_.TotalPixels * sizeof(float)));
}

void RayMarchingTexture::Update()
{
    Param_.MinDistance = Param_.DistanceRatio * fminf(GetDistanceFromCamera(), 1.f);

    if (NextStep_ != RMS::None)
    {
        MarchTimer_.Start();
        Param_.CameraLeft   = Cross(Param_.CameraUp, Param_.CameraDir);
        Param_.CameraRealUp = Cross(Param_.CameraDir, Param_.CameraLeft);
        LaunchRayMarching(Param_, NextStep_);
        CudaCheck(cudaDeviceSynchronize());
        MarchTimer_.Stop();

        switch (NextStep_)
        {
        case RMS::HalfRes:
            NextStep_             = RMS::FillRes;
            Param_.CurrentSubstep = 1;
            break;
        case RMS::FillRes:
            Param_.CurrentSubstep++;
            if (Param_.CurrentSubstep > 3)
            {
                NextStep_             = RMS::None;
                Param_.CurrentSubstep = 0;
            }
            break;
        case RMS::FullRes: NextStep_ = RMS::None; break;
        default: break;
        }

        MapBuffers();

        CopyTimer_.Start();
        CudaCheck(cudaMemcpyToArray(TexArray_, 0, 0, Param_.TexCuda, TexDataSize_, cudaMemcpyDeviceToDevice));
        CopyTimer_.Stop();

        UnmapBuffers();
    }

    {
        if (MarchTimer_.GetCount() >= 60)
        {
            cout << setprecision(3) << "Average time: " << MarchTimer_.GetAverageTimeMs() << " ms" << endl;
            cout << setprecision(3) << "   copy time: " << CopyTimer_.GetAverageTimeMs() << " ms" << endl;
            MarchTimer_.Reset();
            CopyTimer_.Reset();
        }
    }

    {
        static bool first = true;
        if (first)
        {
            first = false;
            ResetView();
        }
    }
}

void RayMarchingTexture::SetCameraInfo(const float3 Position, const float3 Direction, const float3 Up)
{
    Param_.CameraPos = Position;
    Param_.CameraDir = Direction;
    Param_.CameraUp  = Up;
}

float RayMarchingTexture::GetDistanceFromCamera()
{
    return GetDistanceFromPos(Param_.CameraPos);
}

void RayMarchingTexture::SetPerspective(float FOVy, float /*AspectRatio*/, float /*zNear*/, float zFar)
{
    Param_.Depth  = zFar;
    Param_.Height = tanf(FOVy * 3.14159265f / 180.f / 2.f) * Param_.Depth * 2.f;
    //    Param_.Width  = Param_.Height * AspectRatio;
    Param_.Width = Param_.Height * (float)Param_.Size.x / Param_.Size.y;
}

void RayMarchingTexture::IncreaseMaxSteps()
{
    if (Param_.MaxSteps < numeric_limits<int>::max()) Param_.MaxSteps++;
}

void RayMarchingTexture::DecreaseMaxSteps()
{
    if (Param_.MaxSteps > 1) Param_.MaxSteps--;
}

void RayMarchingTexture::IncreaseMinDist()
{
    if (Param_.DistanceRatio < numeric_limits<float>::max() / (INC_FACTOR + 1.f)) Param_.DistanceRatio *= INC_FACTOR;
}

void RayMarchingTexture::DecreaseMinDist()
{
    if (Param_.DistanceRatio > 0.f) Param_.DistanceRatio *= 1.f / INC_FACTOR;
}

void RayMarchingTexture::PrintMarchingParam()
{
    cout << "Min distance = " << Param_.MinDistance << endl << "Max steps = " << Param_.MaxSteps << endl;
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
