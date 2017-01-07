#ifndef FRACTALOBJECT_H
#define FRACTALOBJECT_H

#include "Common.h"
#include "DisplayItem.h"
#include "HostDeviceCode.h"

class FractalObject : public DisplayItem
{
public:
    FractalObject();

    void Init(ShaderProgram* Shaders, TransformMatrix* Projection, TransformMatrix* Visualization,
              const float3* Vertices, const float3* Normals, const uint NumVertices,
              const uint3* Connectivity, const uint NumElements,
              const float4* Colors, const uint NumColors) override;

private:
    cudaGraphicsResource** CudaVboResources_;

    uint MaxTriangles_;

    ArrayPointers DevicePointers_;

    void InitCuda();
    void MapBuffers();
    void UnmapBuffers();
    void** GetArrayAddress(uint BufferIndex);
};

#endif // FRACTALOBJECT_H
