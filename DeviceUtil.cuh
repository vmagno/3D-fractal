#ifndef DEVICEUTIL_CUH
#define DEVICEUTIL_CUH

#include "CudaMath.h"
#include "HostDeviceCode.h"

__device__ uint GetGlobalThreadId()
{
    const uint IdInBlock = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const uint BlockId = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    const uint GlobalThreadId = BlockId * blockDim.z * blockDim.y * blockDim.x + IdInBlock;

    return GlobalThreadId;
}

__device__ uint3 GetVoxelLocation(const KernelParameters& Param, uint VoxelId)
{
    uint3 Location;
    Location.x = VoxelId % Param.VoxelGridSize.x;
    Location.y = (VoxelId / Param.VoxelGridSize.x) % Param.VoxelGridSize.y;
    uint Tmp = Param.VoxelGridSize.x * Param.VoxelGridSize.y;
    Location.z = (VoxelId / Tmp) % Tmp;

    return Location;
}

__device__ uint GetVoxelIdFromLocation(const KernelParameters& Param, const uint3& VoxelLocation)
{
    if (
            VoxelLocation.x < Param.VoxelGridSize.x
            && VoxelLocation.y < Param.VoxelGridSize.y
            && VoxelLocation.z < Param.VoxelGridSize.z)
    {
        return ((VoxelLocation.z * Param.VoxelGridSize.y) + VoxelLocation.y) * Param.VoxelGridSize.x + VoxelLocation.x;
    }

    return Param.NumVoxels;
}

__device__ float GetVoxelValue(const ArrayPointers& DevicePointers, uint VoxelId)
{
    return DevicePointers.VoxelValues[VoxelId] / 255.f;
}

__device__ float GetVoxelValue(const KernelParameters& Param, const ArrayPointers& DevicePointers, const uint3& VoxelLocation)
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
__device__ void SetVoxelValue(float NewValue, const uint3& VoxelLocation, const KernelParameters& Param, const ArrayPointers& DevicePointers)
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

__device__ float3 InterpolateVertex(const float3& VertexA, const float3& VertexB, float ValueA, float ValueB, float Alpha)
{
    const float t = fmaxf(0.f, fminf(1.f, (Alpha - ValueA) / (ValueB - ValueA)));
    return VertexA + ((VertexB - VertexA) * t);
}

__device__ float3 GetNextIter(const float3& Pos, uint Power)
{
    float Rho = Length(Pos);
    float Phi = atan2f(Pos.y, Pos.x);
    float Theta = atan2f(sqrtf(Pos.x*Pos.x + Pos.y*Pos.y), Pos.z);

    float3 Next =
            Pos +
            make_float3(sinf(Power * Theta) * cosf(Power * Phi),
                        sinf(Power * Theta) * sinf(Power * Phi),
                        cosf(Power * Theta))
            * powf(Rho, Power);

    return Next;
}

__device__ bool IsInMandelbulb(const float3& Position, uint Power, uint NumIterations)
{
    const float Threshold = 0.001f;
    float3 CurrentValue = Position;
    float CurrentLength = Length(Position);

    for (int i = 0; i < NumIterations; i++)
    {
        const float3 NextValue = GetNextIter(CurrentValue, Power);

        const float NextLength = Length(NextValue);
        if (fabsf(NextLength - CurrentLength) < Threshold) return true;

        CurrentLength = NextLength;
        CurrentValue = NextValue;
    }

    return false;
}

__device__ uint MakeColor(uchar R, uchar G, uchar B, uchar A)
{
    return A << 24 | B << 16 | G << 8 | R;
}

#endif // DEVICEUTIL_CUH
