#ifndef FRACTALOBJECT_H
#define FRACTALOBJECT_H

#include "Common.h"
#include "DisplayItem.h"
#include "HostDeviceCode.h"

class FractalObject : public DisplayItem
{
public:
    FractalObject();
    ~FractalObject();

    void Init(ShaderProgram* Shaders, TransformMatrix* Projection, TransformMatrix* Visualization,
              const float3* Vertices, const float3* Normals, const uint NumVertices,
              const uint3* Connectivity, const uint NumElements,
              const float4* Colors, const uint NumColors,
              const SDL_Surface* TexImage, const float2* TexCoord) override;

    void Update();

    void AttachGLTexture(GLuint TextureId, uint TexDataSize);

    void MovePlane(int Distance);

private:
    cudaGraphicsResource** CudaVboResources_;

    cudaGraphicsResource* TexResource_;
    cudaArray* TexArray_;
//    uint* TexCudaTarget_;
    uint TexDataSize_;

    uint MaxTriangles_;

    KernelParameters Param_;
    ArrayPointers DevicePointers_;

    void InitTexture();
    void InitCuda();
    void MapBuffers();
    void UnmapBuffers();
    void** GetArrayAddress(uint BufferIndex);
    void CreateTextures();
    void LoadVoxels();
};

#endif // FRACTALOBJECT_H
