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
    return ((VoxelLocation.z * Param.VoxelGridSize.y) + VoxelLocation.y) * Param.VoxelGridSize.x + VoxelLocation.x;
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

__device__ bool IsInMandelbulb(const float3& Position, uint Power, uint NumIterations)
{
    const float Threshold = 0.01f;
    float3 CurrentValue = Position;
    float CurrentLength = Length(Position);
    float PreviousLength = 0.f;

    for (int i = 0; i < NumIterations; i++)
    {
        float Rho = Length(CurrentValue);
        float Phi = atanf(CurrentValue.y / CurrentValue.x);
        float Theta = atanf(sqrtf(CurrentValue.x*CurrentValue.x + CurrentValue.y*CurrentValue.y) / CurrentValue.z);

        CurrentValue =
                make_float3(sin(Power * Theta) * cos(Power * Phi),
                            sin(Power * Theta) * sin(Power * Phi),
                            cos(Power * Theta))
                * Rho;

        CurrentLength = Length(CurrentValue);
        if (CurrentLength - PreviousLength < Threshold) return true;

        PreviousLength = CurrentLength;
    }

    return false;
}

#endif // DEVICEUTIL_CUH
