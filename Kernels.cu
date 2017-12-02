#include "Kernels.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include "CudaMath.h"
#include "DeviceUtil.cuh"
#include "DistanceEstimators.cuh"

using namespace DeviceUtilities;
using namespace DistanceEstimation;
using DE = DEType;

const DE DistFunction = DE::FractalTriangle;

__global__ void TestKernel(KernelParameters Param, ArrayPointers DevicePointers)
{
    uint Tid = GetGlobalThreadId();

    if (Tid >= 4) return;

    switch (Tid)
    {
    case 0:
        DevicePointers.Vertices[0]     = make_float3(-2.f, 0.f, 0.f);
        DevicePointers.Connectivity[0] = make_uint3(0, 1, 2);
        DevicePointers.Normals[0]      = make_float3(-1.f, 0.f, -0.1f);
        DevicePointers.Colors[0]       = make_float4(1.f, 0.f, 0.f, 1.f);
        break;
    case 1:
        DevicePointers.Vertices[1]     = make_float3(-1.f, 0.f, 0.f);
        DevicePointers.Connectivity[1] = make_uint3(0, 1, 3);
        DevicePointers.Normals[1]      = make_float3(1.f, 0.f, -0.1f);
        DevicePointers.Colors[1]       = make_float4(0.f, 1.f, 0.f, 1.f);
        break;
    case 2:
        DevicePointers.Vertices[2]     = make_float3(-2.f, 1.f, 0.f);
        DevicePointers.Connectivity[2] = make_uint3(1, 2, 3);
        DevicePointers.Normals[2]      = make_float3(0.f, 1.f, -0.1f);
        DevicePointers.Colors[2]       = make_float4(0.f, 0.f, 1.f, 1.f);
        break;
    case 3:
        DevicePointers.Vertices[3]     = make_float3(-1.5f, 0.5f, 1.f);
        DevicePointers.Connectivity[3] = make_uint3(2, 0, 3);
        DevicePointers.Normals[3]      = make_float3(-.1f, 0.f, 1.f);
        DevicePointers.Colors[3]       = make_float4(1.f, 1.f, 1.f, 1.f);
        break;
    default: break;
    }
}

void LaunchTest(const KernelParameters& Param, const ArrayPointers& DevicePointers)
{
    TestKernel<<<1, 4>>>(Param, DevicePointers);

    fflush(stdout);
}

__global__ void ClassifyVoxels(KernelParameters Param, ArrayPointers DevicePointers)
{
    uint VoxelId = GetGlobalThreadId();

    if (VoxelId >= Param.NumVoxels) return;

    uint3 VoxelLocation = GetVoxelLocation(Param, VoxelId);

    // Retrieve voxel values in that cell
    float CellValues[8];
    CellValues[0] = GetVoxelValue(DevicePointers, VoxelId);
    CellValues[1] = GetVoxelValue(Param, DevicePointers, VoxelLocation + make_uint3(1, 0, 0));
    CellValues[2] = GetVoxelValue(Param, DevicePointers, VoxelLocation + make_uint3(1, 1, 0));
    CellValues[3] = GetVoxelValue(Param, DevicePointers, VoxelLocation + make_uint3(0, 1, 0));
    CellValues[4] = GetVoxelValue(Param, DevicePointers, VoxelLocation + make_uint3(0, 0, 1));
    CellValues[5] = GetVoxelValue(Param, DevicePointers, VoxelLocation + make_uint3(1, 0, 1));
    CellValues[6] = GetVoxelValue(Param, DevicePointers, VoxelLocation + make_uint3(1, 1, 1));
    CellValues[7] = GetVoxelValue(Param, DevicePointers, VoxelLocation + make_uint3(0, 1, 1));

    // Set texture color (for debugging)
    if (VoxelLocation.z == Param.ZSlice)
    {
        float Val = CellValues[0]; //(VoxelLocation.x + VoxelLocation.y) / (31.f + 31.f);
        DevicePointers
          .TexCudaTarget[(Param.VoxelGridSize.y - VoxelLocation.y - 1) * Param.VoxelGridSize.x + VoxelLocation.x] =
          MakeColor(Val * 255, 0, 255 - uint(Val * 255) / 32 * 31.874, 0xff);
    }

    // Compute cell index
    uint CellId = 0;
    for (int i = 0; i < 8; i++)
    {
        CellId += uint(CellValues[i] < Param.Threshold) << i;
    }

    // Retrieve number of vertices in that cell
    DevicePointers.VoxelVertices[VoxelId] = tex1Dfetch<uint>(DevicePointers.NumVertexTableTex, CellId);
}

void LaunchClassifyVoxel(const KernelParameters& Param, const ArrayPointers& DevicePointers)
{
    ClassifyVoxels<<<Param.NumBlocks, Param.BlockSize>>>(Param, DevicePointers);
    fflush(stdout);
}

void LaunchThrustScan(KernelParameters Param, ArrayPointers DevicePointers)
{
    thrust::exclusive_scan(thrust::device_ptr<uint>(DevicePointers.VoxelVertices),
                           thrust::device_ptr<uint>(DevicePointers.VoxelVertices + Param.NumVoxels),
                           thrust::device_ptr<uint>(DevicePointers.VoxelVerticesScan));
}

__global__ void GenerateTriangles(KernelParameters Param, ArrayPointers DevicePointers)
{
    const uint VoxelId = GetGlobalThreadId();

    if (VoxelId >= Param.NumVoxels) return;

    const uint3 VoxelLocation = GetVoxelLocation(Param, VoxelId);

    // Compute cube vertices
    float3 Positions[8];
    Positions[0] =
      Param.MinPosition + make_float3(VoxelLocation.x * Param.VoxelSize.x, VoxelLocation.y * Param.VoxelSize.y,
                                      VoxelLocation.z * Param.VoxelSize.z);
    Positions[1] = Positions[0] + make_float3(Param.VoxelSize.x, 0.f, 0.f);
    Positions[2] = Positions[0] + make_float3(Param.VoxelSize.x, Param.VoxelSize.y, 0.f);
    Positions[3] = Positions[0] + make_float3(0.f, Param.VoxelSize.y, 0.f);
    Positions[4] = Positions[0] + make_float3(0.f, 0.f, Param.VoxelSize.z);
    Positions[5] = Positions[0] + make_float3(Param.VoxelSize.x, 0.f, Param.VoxelSize.z);
    Positions[6] = Positions[0] + make_float3(Param.VoxelSize.x, Param.VoxelSize.y, Param.VoxelSize.z);
    Positions[7] = Positions[0] + make_float3(0.f, Param.VoxelSize.y, Param.VoxelSize.z);

    // Retrieve cell values
    float CellValues[8];
    CellValues[0] = GetVoxelValue(DevicePointers, VoxelId);
    CellValues[1] = GetVoxelValue(Param, DevicePointers, VoxelLocation + make_uint3(1, 0, 0));
    CellValues[2] = GetVoxelValue(Param, DevicePointers, VoxelLocation + make_uint3(1, 1, 0));
    CellValues[3] = GetVoxelValue(Param, DevicePointers, VoxelLocation + make_uint3(0, 1, 0));
    CellValues[4] = GetVoxelValue(Param, DevicePointers, VoxelLocation + make_uint3(0, 0, 1));
    CellValues[5] = GetVoxelValue(Param, DevicePointers, VoxelLocation + make_uint3(1, 0, 1));
    CellValues[6] = GetVoxelValue(Param, DevicePointers, VoxelLocation + make_uint3(1, 1, 1));
    CellValues[7] = GetVoxelValue(Param, DevicePointers, VoxelLocation + make_uint3(0, 1, 1));

    // Compute cell index
    uint CellId = 0;
    for (int i = 0; i < 8; i++)
    {
        CellId += uint(CellValues[i] < Param.Threshold) << i;
    }

    // Compute interpolated vertex positions
    float3 CellVertices[12];
    CellVertices[0] = InterpolateVertex(Positions[0], Positions[1], CellValues[0], CellValues[1], Param.Threshold);
    CellVertices[1] = InterpolateVertex(Positions[1], Positions[2], CellValues[1], CellValues[2], Param.Threshold);
    CellVertices[2] = InterpolateVertex(Positions[2], Positions[3], CellValues[2], CellValues[3], Param.Threshold);
    CellVertices[3] = InterpolateVertex(Positions[3], Positions[0], CellValues[3], CellValues[0], Param.Threshold);

    CellVertices[4] = InterpolateVertex(Positions[4], Positions[5], CellValues[4], CellValues[5], Param.Threshold);
    CellVertices[5] = InterpolateVertex(Positions[5], Positions[6], CellValues[5], CellValues[6], Param.Threshold);
    CellVertices[6] = InterpolateVertex(Positions[6], Positions[7], CellValues[6], CellValues[7], Param.Threshold);
    CellVertices[7] = InterpolateVertex(Positions[7], Positions[4], CellValues[7], CellValues[4], Param.Threshold);

    CellVertices[8]  = InterpolateVertex(Positions[0], Positions[4], CellValues[0], CellValues[4], Param.Threshold);
    CellVertices[9]  = InterpolateVertex(Positions[1], Positions[5], CellValues[1], CellValues[5], Param.Threshold);
    CellVertices[10] = InterpolateVertex(Positions[2], Positions[6], CellValues[2], CellValues[6], Param.Threshold);
    CellVertices[11] = InterpolateVertex(Positions[3], Positions[7], CellValues[3], CellValues[7], Param.Threshold);

    // Form the triangles
    const uint NumVertices = tex1Dfetch<uint>(DevicePointers.NumVertexTableTex, CellId);

    for (uint iVert = 0; iVert < NumVertices; iVert += 3)
    {
        uint Index = DevicePointers.VoxelVerticesScan[VoxelId] + iVert;

        float3* TriangleVertices[3];
        for (int j = 0; j < 3; j++)
        {
            uint Edge           = tex1Dfetch<uint>(DevicePointers.TriangleTableTex, CellId * 16 + iVert + j);
            TriangleVertices[j] = &CellVertices[Edge];
        }

        float3 Normal = CalculateNormal(*TriangleVertices[0], *TriangleVertices[1], *TriangleVertices[2]);

        if (Index < (Param.MaxVertices - 3))
        {
            DevicePointers.Vertices[Index]     = *TriangleVertices[0];
            DevicePointers.Vertices[Index + 1] = *TriangleVertices[1];
            DevicePointers.Vertices[Index + 2] = *TriangleVertices[2];

            DevicePointers.Normals[Index]     = Normal;
            DevicePointers.Normals[Index + 1] = Normal;
            DevicePointers.Normals[Index + 2] = Normal;

            DevicePointers.Connectivity[Index / 3] = make_uint3(Index, Index + 1, Index + 2);

            DevicePointers.Colors[Index]     = make_float4(1.f);
            DevicePointers.Colors[Index + 1] = make_float4(1.f);
            DevicePointers.Colors[Index + 2] = make_float4(1.f);
        }
    }
}

void LaunchGenerateTriangles(const KernelParameters& Param, const ArrayPointers& DevicePointers)
{
    GenerateTriangles<<<Param.NumBlocks, Param.BlockSize>>>(Param, DevicePointers);
    fflush(stdout);
}

__global__ void SampleVolume(KernelParameters Param, ArrayPointers DevicePointers)
{
    const uint VoxelId = GetGlobalThreadId();

    if (VoxelId >= Param.NumVoxels) return;

    const uint3  VoxelLocation = GetVoxelLocation(Param, VoxelId);
    const float3 VoxelPosition = Param.MinPosition + (make_float3(VoxelLocation) * Param.VoxelSize);

    float NewValue = 0.f;
    if (IsInMandelbulb(VoxelPosition, 8, 10))
    {
        NewValue = 1.f;
    }

    SetVoxelValue(NewValue, VoxelId, DevicePointers);
}

void LaunchSampleVolume(const KernelParameters& Param, const ArrayPointers& DevicePointers)
{
    SampleVolume<<<Param.NumBlocks, Param.BlockSize>>>(Param, DevicePointers);
    fflush(stdout);
}

template <RMS   State>
__global__ void RayMarching(RayMarchingParam Param)
{
    if (State == RMS::None) return;

    const uint PixelId = GetPixelId<State>(Param, GetGlobalThreadId(), Param.CurrentSubstep);

    if (PixelId >= Param.TotalPixels) return;

    const uint2  Location  = GetPixelLocationFromId(Param, PixelId);
    const float3 Direction = GetRayDirection(Param, Location);

    uint        NumSteps;
    const float Distance = GetTotalDistance<DistFunction>(Param, Direction, NumSteps);

    const float Brightness = 1.f - static_cast<float>(NumSteps) / Param.MaxSteps;
    const uint  Color      = MakeColor((uchar)(255 * Brightness), 0, 0, 0xff);

    Param.TexCuda[PixelId] = Color;
    if (State == RMS::HalfRes)
    {
        Param.TexCuda[PixelId + 1]                = Color;
        Param.TexCuda[PixelId + Param.Size.x]     = Color;
        Param.TexCuda[PixelId + Param.Size.x + 1] = Color;
    }

    Param.Distances[PixelId] = Distance;
    if (State == RMS::HalfRes)
    {
        Param.Distances[PixelId + 1]                = Distance;
        Param.Distances[PixelId + Param.Size.x]     = Distance;
        Param.Distances[PixelId + Param.Size.x + 1] = Distance;
    }
}

#if 0
__global__ void ComputeColor(RayMarchingParam Param)
{
    const uint PixelId = GetPixelId<RMS::FullRes>(Param, GetGlobalThreadId());

    if (PixelId >= Param.TotalPixels) return;
    if (Param.Distances[PixelId] >= Param.DistanceThreshold) return;

    uint Neighbours[8];
    GetNeighbourPixels(Param, PixelId, Neighbours);
    float Distances[4];
    GetDistValues(Param, Neighbours, Distances);

    const float EpsilonX = Param.Width / Param.Size.x * 2.f * Param.Distances[PixelId] / Param.Depth;
    const float EpsilonY = Param.Height / Param.Size.y * 2.f * Param.Distances[PixelId] / Param.Depth;

    const float3 Normal = Param.Distances[PixelId] >= Param.DistanceThreshold
                            ? make_float3(0.f)
                            : Normalize(make_float3((Distances[1] - Distances[0]) / EpsilonX,
                                                    (Distances[3] - Distances[2]) / EpsilonY, 1.0f));

    const uint2 Location = GetPixelLocationFromId(Param, PixelId);

    const float OffsetX = ((float)Location.x - Param.Size.x / 2) / Param.Size.x * Param.Width;
    const float OffsetY = ((float)Location.y - Param.Size.y / 2) / Param.Size.y * Param.Height;

    const float3 Target =
      Param.CameraPos + Param.CameraDir * Param.Depth - OffsetX * Param.CameraLeft - OffsetY * Param.CameraRealUp;
    const float3 Direction      = Normalize(Target - Param.CameraPos);
    const float3 Position       = Direction * Param.Distances[PixelId];
    const float3 LightDirection = Normalize(Position - Param.LightPos);

    const uint Color = MakeColor(Normal.x * 127 + 127, Normal.y * 127 + 127, Normal.z * 127 + 127, 0xff);
    // const uint Color = MakeColor(255 - 255 * Dot(1.f * LightDirection, Normal), 0, 0, 255);

#if 0 // DEBUG
    const uint TargetPixel = Param.TotalPixels / 2 + Param.Size.x / 2 + Param.Size.x * 130;
    if (PixelId == TargetPixel)
    {
        printf("Epsilons = %f %f\n", EpsilonX, EpsilonY);
        printf("Pixel pos = %d %d\n", Location.x, Location.y);
        printf("Distances: mid %f, left %f, right %f, up %f, down %f\n", Param.Distances[PixelId], Distances[0],
               Distances[1], Distances[2], Distances[3]);
        printf("Base normal: %f %f %f\n", (Distances[1] - Distances[0]) / EpsilonX,
               (Distances[3] - Distances[2]) / EpsilonY, EpsilonX * 0.5f);
        printf("Normalized normal: %f %f %f\n", Normal.x, Normal.y, Normal.z);
    }

    if (PixelId == TargetPixel)
    {
        Param.TexCuda[PixelId] = MakeColor(0, 0, 0, 0xff);
        for (int i = 0; i < 8; i++)
        {
            if (Neighbours[i] < Param.TotalPixels)
            {
                Param.TexCuda[Neighbours[i]] = MakeColor(0, 0, 0, 0xff);
            }
        }
    }
#endif

    Param.TexCuda[PixelId] = Color;
}
#else
__global__ void ComputeColor(RayMarchingParam Param)
{
    const uint PixelId = GetPixelId<RMS::FullRes>(Param, GetGlobalThreadId());
    if (PixelId >= Param.TotalPixels) return;

    // Has to be proportional to MinDistance I think... Needs some tweaking in any case
    // const float Epsilon = Param.MinDistance * Param.EpsilonFactor * Param.Width / Param.Size.x * 2.0f *
    // Param.Distances[PixelId] / Param.Depth;
    const float Epsilon = Param.EpsilonFactor / Param.Distances[PixelId];
    //    const float Epsilon = Param.EpsilonFactor;
    // const float Epsilon = 0.005f;
    const uint2 Location = GetPixelLocationFromId(Param, PixelId);

    uint   NumSteps;
    float3 Direction;

    Direction         = GetRayDirection(Param, Location, Epsilon, 0.f);
    const float Xplus = GetTotalDistance<DistFunction>(Param, Direction, NumSteps);

    Direction          = GetRayDirection(Param, Location, -Epsilon, 0.f);
    const float Xminus = GetTotalDistance<DistFunction>(Param, Direction, NumSteps);

    Direction         = GetRayDirection(Param, Location, 0.f, Epsilon);
    const float Yplus = GetTotalDistance<DistFunction>(Param, Direction, NumSteps);

    Direction          = GetRayDirection(Param, Location, 0.f, -Epsilon);
    const float Yminus = GetTotalDistance<DistFunction>(Param, Direction, NumSteps);

    const float3 Normal = Normalize(make_float3((Xplus - Xminus) / 2.f / Epsilon, (Yplus - Yminus) / 2.f / Epsilon,
                                                Param.Distances[PixelId] / Param.Depth));
    //    const float3 Position = GetRayDirection(Param, Location) * Param.Distances[PixelId];
    //    const float3 LightDirection = Normalize(Position - Param.LightPos);

    Direction                   = GetRayDirection(Param, Location);
    const float3 Position       = Direction * Param.Distances[PixelId];
    const float3 LightDirection = Normalize(Position - Param.LightPos);

    // const uint Color = MakeColor(Normal.x * 127 + 127, Normal.y * 127 + 127, Normal.z * 127 + 127, 255);
    const float NdotL = fmaxf(Dot(-1.f * LightDirection, Normal), 0.f);
    const uint  Color = MakeColor(255 - 250 * NdotL, 100, 0, 255);
    // const uint Color = MakeColor(LightDirection.x * 127 + 127, LightDirection.y * 127 + 127, LightDirection.z * 127 +
    // 127, 255);

    if (Param.Distances[PixelId] < Param.DistanceThreshold)
    {
        
        Param.TexCuda[PixelId] *= NdotL;
    }

#if 1 // DEBUG
    const uint TargetPixel = Param.TotalPixels / 2 + Param.Size.x / 2;
    if (PixelId == TargetPixel)
    {
        printf("Epsilon = %f\n", Epsilon);
        printf("Pixel pos = %d %d\n", Location.x, Location.y);
        printf("Distances: mid %f, left %f, right %f, up %f, down %f\n", Param.Distances[PixelId], Xminus, Xplus,
               Yminus, Yplus);
        //        printf("Base normal: %f %f %f\n", (Distances[1] - Distances[0]) / EpsilonX,
        //               (Distances[3] - Distances[2]) / EpsilonY, EpsilonX * 0.5f);
        //        printf("Normalized normal: %f %f %f\n", Normal.x, Normal.y, Normal.z);
    }

    if (PixelId == TargetPixel)
    {
        Param.TexCuda[PixelId]                    = MakeColor(0, 0, 0, 0xff);
        Param.TexCuda[PixelId + 1]                = MakeColor(0, 0, 0, 0xff);
        Param.TexCuda[PixelId - 1]                = MakeColor(0, 0, 0, 0xff);
        Param.TexCuda[PixelId + Param.Size.x]     = MakeColor(0, 0, 0, 0xff);
        Param.TexCuda[PixelId + Param.Size.x + 1] = MakeColor(0, 0, 0, 0xff);
        Param.TexCuda[PixelId + Param.Size.x - 1] = MakeColor(0, 0, 0, 0xff);
        Param.TexCuda[PixelId - Param.Size.x]     = MakeColor(0, 0, 0, 0xff);
        Param.TexCuda[PixelId - Param.Size.x + 1] = MakeColor(0, 0, 0, 0xff);
        Param.TexCuda[PixelId - Param.Size.x - 1] = MakeColor(0, 0, 0, 0xff);
    }
#endif
}
#endif

void LaunchRayMarching(const RayMarchingParam& Param, const RMS Step)
{
    switch (Step)
    {
    case RMS::HalfRes:
    {
        // std::cout << "Half res kernel" << std::endl;
        const uint NumBlocks = (uint)ceilf((float)Param.Size.x * Param.Size.y / Param.BlockSize.x / Param.BlockSize.y /
                                           Param.BlockSize.z / 4);
        RayMarching<RMS::HalfRes><<<NumBlocks, Param.BlockSize>>>(Param);
        break;
    }
    case RMS::FillRes:
    {
        const uint NumBlocks = (uint)ceilf((float)Param.Size.x * Param.Size.y / Param.BlockSize.x / Param.BlockSize.y /
                                           Param.BlockSize.z / 4);
        // std::cout << "Fill res kernel, substep " << Param.CurrentSubstep << ", NumBlocks " << NumBlocks << std::endl;
        RayMarching<RMS::FillRes><<<NumBlocks, Param.BlockSize>>>(Param);
        break;
    }
    case RMS::FullRes:
        // std::cout << "Full res kernel" << std::endl;
        RayMarching<RMS::FullRes><<<Param.NumBlocks, Param.BlockSize>>>(Param);
        break;
    case RMS::ComputeColor:
        // std::cout << "Compute color kernel" << std::endl;
        ComputeColor<<<Param.NumBlocks, Param.BlockSize>>>(Param);
        break;
    case RMS::None:
        // std::cout << "Some other RMS" << std::endl;
        break;
    }
}

float GetDistanceFromPos(const float3& Position)
{
    return GetDistance<DistFunction>(Position);
}

template <typename T>
__global__ void InitValueKernel(T* Vector, size_t Size, T Value)
{
    uint Index = GetGlobalThreadId();
    if (Index >= Size) return;
    Vector[Index] = Value;
}

template <typename T>
void InitCaller::InitValue(T* Vector, size_t Size, T Value)
{
    const uint NumThreadsPerBlock = 512;
    const dim3 BlockSize(NumThreadsPerBlock);
    const dim3 GridSize((uint)ceilf((float)Size / NumThreadsPerBlock));
    InitValueKernel<<<GridSize, BlockSize>>>(Vector, Size, Value);
}

template void InitCaller::InitValue<uint>(uint*, size_t, uint);
template void InitCaller::InitValue<float>(float*, size_t, float);
