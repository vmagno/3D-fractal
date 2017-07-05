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

const DEType DistFunction = FractalTriangle;

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

template <RayMarchingStep State>
__global__ void RayMarching(RayMarchingParam Param)
{
    if (State == None) return;

    const uint PixelId = GetPixelId<State>(Param, GetGlobalThreadId(), Param.CurrentSubstep);
    //    const uint PixelId = GetPixelId<FullRes>(Param, GetGlobalThreadId());

    if (PixelId >= Param.TotalPixels) return;

    const uint PixelPosX = PixelId % Param.Size.x;
    const uint PixelPosY = PixelId / Param.Size.y;

    const float OffsetX = ((float)PixelPosX - Param.Size.x / 2) / Param.Size.x * Param.Width;
    const float OffsetY = ((float)PixelPosY - Param.Size.y / 2) / Param.Size.y * Param.Height;

    const float3 Target =
      Param.CameraPos + Param.CameraDir * Param.Depth - OffsetX * Param.CameraLeft - OffsetY * Param.CameraRealUp;
    const float3 Direction = Normalize(Target - Param.CameraPos);

    float TotalDist = 0.f;
    uint  steps;
    for (steps = 0; steps < Param.MaxSteps; steps++)
    {
        const float3 Position = Param.CameraPos + TotalDist * Direction;
        const float  Distance = GetDistance<DistFunction>(Position);
        TotalDist += Distance;
        if (Distance < Param.MinDistance) break;
    }

    const float Brightness = 1.f - static_cast<float>(steps) / Param.MaxSteps;
    const uint  Color      = MakeColor((uchar)(255 * Brightness), 0, 0, 0xff);

    Param.TexCuda[PixelId] = Color;
    if (State == HalfRes)
    {
        Param.TexCuda[PixelId + 1]                = Color;
        Param.TexCuda[PixelId + Param.Size.x]     = Color;
        Param.TexCuda[PixelId + Param.Size.x + 1] = Color;
    }
}

void LaunchRayMarching(const RayMarchingParam& Param, const RayMarchingStep Step)
{
    switch (Step)
    {
    case HalfRes:
    {
        //        std::cout << "Half res kernel" << std::endl;
        const uint NumBlocks = (uint)ceilf((float)Param.Size.x * Param.Size.y / Param.BlockSize.x / Param.BlockSize.y /
                                           Param.BlockSize.z / 4);
        RayMarching<HalfRes><<<NumBlocks, Param.BlockSize>>>(Param);
        break;
    }
    case FillRes:
    {
        //        std::cout << "Fill res kernel" << std::endl;
        const uint NumBlocks = (uint)ceilf((float)Param.Size.x * Param.Size.y / Param.BlockSize.x / Param.BlockSize.y /
                                           Param.BlockSize.z / 4);
        RayMarching<FillRes><<<NumBlocks, Param.BlockSize>>>(Param);
        break;
    }
    case FullRes:
        //        std::cout << "Full res kernel" << std::endl;
        RayMarching<FullRes><<<Param.NumBlocks, Param.BlockSize>>>(Param);
        break;
    case None: break;
    }
}

float GetDistanceFromPos(const float3& Position)
{
    return GetDistance<DistFunction>(Position);
}
