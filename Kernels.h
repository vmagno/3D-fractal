#ifndef KERNELS_H
#define KERNELS_H

#include "Common.h"
#include "HostDeviceCode.h"

extern void LaunchTest(const KernelParameters& Param, const ArrayPointers& DevicePointers);

extern void LaunchClassifyVoxel(const KernelParameters& Param, const ArrayPointers& DevicePointers);
extern void LaunchThrustScan(KernelParameters Param, ArrayPointers DevicePointers);
extern void LaunchGenerateTriangles(const KernelParameters& Param, const ArrayPointers& DevicePointers);

extern void LaunchSampleVolume(const KernelParameters& Param, const ArrayPointers& DevicePointers);

extern void LaunchRayMarching(const RayMarchingParam& Param);
extern float GetDistanceFromPos(const float3& Position);

#endif // KERNELS_H
