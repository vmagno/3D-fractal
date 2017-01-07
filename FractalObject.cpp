#include "FractalObject.h"

#include <cuda_gl_interop.h>

using namespace std;

FractalObject::FractalObject()
    : DisplayItem(),
      CudaVboResources_(NULL),
      MaxTriangles_(0)
{
    DevicePointers_.Vertices = NULL;
    DevicePointers_.Normals = NULL;
    DevicePointers_.Connectivity = NULL;
    DevicePointers_.Colors = NULL;
}

void FractalObject::Init(ShaderProgram* Shaders, TransformMatrix* Projection, TransformMatrix* Visualization,
                         const float3* /*Vertices*/, const float3* /*Normals*/, const uint NumVertices,
                         const uint3* /*Connectivity*/, const uint NumElements,
                         const float4* /*Colors*/, const uint /*NumColors*/)
{
    DisplayItem::Init(Shaders, Projection, Visualization, NULL, NULL, NumVertices, NULL, NumElements, NULL, NumVertices);

    MaxTriangles_ = NumElements;

    InitCuda();
}

void FractalObject::InitCuda()
{
    CudaVboResources_ = new cudaGraphicsResource*[NumVBO_];

    for (uint iBuffer = 0; iBuffer < NumVBO_; iBuffer++)
    {
        CudaCheck(cudaGraphicsGLRegisterBuffer(&CudaVboResources_[iBuffer], VBOs_[iBuffer], cudaGraphicsMapFlagsNone));
    }

    MapBuffers();
    UnmapBuffers();
}

void FractalObject::MapBuffers()
{
    for (uint iBuffer = 0; iBuffer < NumVBO_; iBuffer++)
    {
        size_t NumBytes;
        CudaCheck(cudaGraphicsMapResources(1, &CudaVboResources_[iBuffer], 0));
        CudaCheck(cudaGraphicsResourceGetMappedPointer(GetArrayAddress(iBuffer), &NumBytes, CudaVboResources_[iBuffer]));
    }
}

void FractalObject::UnmapBuffers()
{
    for (uint iBuffer = 0; iBuffer < NumVBO_; iBuffer++)
    {
        CudaCheck(cudaGraphicsUnmapResources(1, &CudaVboResources_[iBuffer]));
    }
}

void** FractalObject::GetArrayAddress(uint BufferIndex)
{
    switch (BufferIndex) {
    case VERTEX_VBO_ID:
        return (void**)&DevicePointers_.Vertices;
        break;
    case NORMAL_VBO_ID:
        return (void**)&DevicePointers_.Normals;
        break;
    case CONNECT_VBO_ID:
        return (void**)&DevicePointers_.Connectivity;
        break;
    case COLOR_VBO_ID:
        return (void**)&DevicePointers_.Colors;
        break;
    default:
        cerr << "[ERROR] Accessing a non-existing buffer (trying to map VBOs to CUDA resources)" << endl;
        break;
    }

    return NULL;
}
