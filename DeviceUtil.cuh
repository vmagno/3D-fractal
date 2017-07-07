#ifndef DEVICEUTIL_CUH
#define DEVICEUTIL_CUH

#include <vector>

#include "CudaMath.h"
#include "HostDeviceCode.h"

namespace DeviceUtilities {

using RMS = RayMarchingStep;

__device__ uint GetGlobalThreadId()
{
    const uint IdInBlock      = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const uint BlockId        = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    const uint GlobalThreadId = BlockId * blockDim.z * blockDim.y * blockDim.x + IdInBlock;

    return GlobalThreadId;
}

__device__ uint3 GetVoxelLocation(const KernelParameters& Param, uint VoxelId)
{
    uint3 Location;
    Location.x = VoxelId % Param.VoxelGridSize.x;
    Location.y = (VoxelId / Param.VoxelGridSize.x) % Param.VoxelGridSize.y;
    uint Tmp   = Param.VoxelGridSize.x * Param.VoxelGridSize.y;
    Location.z = (VoxelId / Tmp) % Tmp;

    return Location;
}

__device__ uint GetVoxelIdFromLocation(const KernelParameters& Param, const uint3& VoxelLocation)
{
    if (VoxelLocation.x < Param.VoxelGridSize.x && VoxelLocation.y < Param.VoxelGridSize.y &&
        VoxelLocation.z < Param.VoxelGridSize.z)
    {
        return ((VoxelLocation.z * Param.VoxelGridSize.y) + VoxelLocation.y) * Param.VoxelGridSize.x + VoxelLocation.x;
    }

    return Param.NumVoxels;
}

__device__ float GetVoxelValue(const ArrayPointers& DevicePointers, uint VoxelId)
{
    return DevicePointers.VoxelValues[VoxelId] / 255.f;
}

__device__ float GetVoxelValue(const KernelParameters& Param, const ArrayPointers& DevicePointers,
                               const uint3& VoxelLocation)
{
    const uint VoxelId = GetVoxelIdFromLocation(Param, VoxelLocation);
    if (VoxelId >= Param.NumVoxels) return 0.f;
    return GetVoxelValue(DevicePointers, VoxelId);
}

/**
 * @brief SetVoxelValue
 * @param NewValue Must be between 0.0f and 0.1f
 * @param VoxelId
 * @param DevicePointers
 */
__device__ void SetVoxelValue(float NewValue, uint VoxelId, const ArrayPointers& DevicePointers)
{
    DevicePointers.VoxelValues[VoxelId] = NewValue * 255.f;
}

/**
 * @brief SetVoxelValue
 * @param NewValue Must be between 0.0f and 0.1f
 * @param VoxelLocation
 * @param Param
 * @param DevicePointers
 */
__device__ void SetVoxelValue(float NewValue, const uint3& VoxelLocation, const KernelParameters& Param,
                              const ArrayPointers& DevicePointers)
{
    const uint VoxelId = GetVoxelIdFromLocation(Param, VoxelLocation);
    if (VoxelId >= Param.NumVoxels) return;
    SetVoxelValue(NewValue, VoxelId, DevicePointers);
}

__device__ float3 CalculateNormal(const float3& V0, const float3& V1, const float3& V2)
{
    const float3 Edge0 = V1 - V0;
    const float3 Edge1 = V2 - V0;
    return Cross(Edge0, Edge1);
}

__device__ float3 InterpolateVertex(const float3& VertexA, const float3& VertexB, float ValueA, float ValueB,
                                    float Alpha)
{
    const float t = fmaxf(0.f, fminf(1.f, (Alpha - ValueA) / (ValueB - ValueA)));
    return VertexA + ((VertexB - VertexA) * t);
}

__device__ float3 GetNextIter(const float3& Pos, uint Power)
{
    float Rho   = Length(Pos);
    float Phi   = atan2f(Pos.y, Pos.x);
    float Theta = atan2f(sqrtf(Pos.x * Pos.x + Pos.y * Pos.y), Pos.z);

    float3 Next = Pos +
                  make_float3(sinf(Power * Theta) * cosf(Power * Phi), sinf(Power * Theta) * sinf(Power * Phi),
                              cosf(Power * Theta)) *
                    powf(Rho, Power);

    return Next;
}

__device__ bool IsInMandelbulb(const float3& Position, uint Power, uint NumIterations)
{
    const float Threshold     = 0.001f;
    float3      CurrentValue  = Position;
    float       CurrentLength = Length(Position);

    for (int i = 0; i < NumIterations; i++)
    {
        const float3 NextValue = GetNextIter(CurrentValue, Power);

        const float NextLength = Length(NextValue);
        if (fabsf(NextLength - CurrentLength) < Threshold) return true;

        CurrentLength = NextLength;
        CurrentValue  = NextValue;
    }

    return false;
}

__device__ uint MakeColor(uchar R, uchar G, uchar B, uchar A)
{
    return A << 24 | B << 16 | G << 8 | R;
}

__device__ uint MakeColor(const float3& Col)
{
    const float3 Tmp = Normalize(Col);
    return MakeColor((uchar)(255 * Tmp.x), (uchar)(255 * Tmp.y), (uchar)(255 * Tmp.z), 255);
}

template <RayMarchingStep State>
__device__ uint GetPixelId(const RayMarchingParam& Param, uint ThreadId, uint CurrentSubstep = 0)
{
    if (State == RMS::HalfRes || State == RMS::FillRes)
    {
        const uint BasePixelId = 2 * (((ThreadId * 2) / Param.Size.x) * Param.Size.x + ThreadId % (Param.Size.x / 2));
        if (State == RMS::HalfRes)
        {
            return BasePixelId;
        }
        else
        {
            if (CurrentSubstep == 1)
                return BasePixelId + 1;
            else if (CurrentSubstep == 2)
                return BasePixelId + Param.Size.x;
            else if (CurrentSubstep == 3)
                return BasePixelId + Param.Size.x + 1;
        }
    }

    return ThreadId;
}

__device__ uint2 GetPixelLocationFromId(const RayMarchingParam& Param, const uint PixelId)
{
    uint2 Location;
    Location.x = PixelId % Param.Size.x;
    Location.y = PixelId / Param.Size.x;
    return Location;
}

__device__ void GetNeighbourPixels(const RayMarchingParam& Param, const uint PixelId, uint (&Neighbours)[8])
{
    uint2 PixelLocation = GetPixelLocationFromId(Param, PixelId);

    for (int i = 0; i < 8; i++)
    {
        Neighbours[i] = Param.TotalPixels;
    }

    if (PixelLocation.y > 0)
    {
        if (PixelLocation.x > 0) Neighbours[0]                = PixelId - Param.Size.x - 1;
        Neighbours[1]                                         = PixelId - Param.Size.x;
        if (PixelLocation.x < Param.Size.x - 1) Neighbours[2] = PixelId - Param.Size.x + 1;
    }

    if (PixelLocation.x > 0) Neighbours[3]                = PixelId - 1;
    if (PixelLocation.x < Param.Size.x - 1) Neighbours[4] = PixelId + 1;

    if (PixelLocation.y < Param.Size.y - 1)
    {
        if (PixelLocation.x > 0) Neighbours[5]                = PixelId + Param.Size.x - 1;
        Neighbours[6]                                         = PixelId + Param.Size.x;
        if (PixelLocation.x < Param.Size.x - 1) Neighbours[7] = PixelId + Param.Size.x + 1;
    }
}

__device__ void GetDistValues(const RayMarchingParam& Param, const uint (&Neighbours)[8], float (&Distances)[4])
{
    uint Lists[4][3] = {{0, 3, 5}, {2, 4, 7}, {0, 1, 2}, {5, 6, 7}};
    for (uint i = 0; i < 4; i++)
    {
        float TmpDist  = 0.f;
        uint  TmpCount = 0;
        for (uint j : Lists[i])
        {
            if (Neighbours[j] < Param.TotalPixels)
            {
                TmpDist += Param.Distances[Neighbours[j]];
                TmpCount++;
            }
        }
        Distances[i] = TmpDist / TmpCount;
    }
}

} // DeviceUtilities

#endif // DEVICEUTIL_CUH
