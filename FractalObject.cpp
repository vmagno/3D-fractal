#include "FractalObject.h"

#include <fstream>
#include <iostream>

#include <cuda_gl_interop.h>
#include <SDL2/SDL.h>

#include "Kernels.h"
#include "MarchingCubesTables.h"
#include "Util.h"

using namespace std;

const uint3 DefaultBlockSize = make_uint3(4, 4, 4);
const uint3 DefaultNumBlocks = make_uint3(0, 0, 0);
const uint3 DefaultVoxelGridSize = make_uint3(128, 128, 128);
//const float3 DefaultMinPosition = make_float3(-0.64f, -0.64f, -0.64f);
const float3 DefaultVoxelSize = make_float3(0.015f, 0.015f, 0.015f);

const float DefaultThreshold = 0.5f;


FractalObject::FractalObject()
    : DisplayItem(),
      CudaVboResources_(NULL),
      TexResource_(NULL),
      TexArray_(NULL),
//      TexCudaTarget_(NULL),
      TexDataSize_(0),
      MaxTriangles_(0)
{
    DevicePointers_.Nullify();

    Param_.BlockSize = DefaultBlockSize;
    Param_.NumBlocks = DefaultNumBlocks;
    Param_.VoxelGridSize = DefaultVoxelGridSize;

    Param_.NumBlocks.x = Param_.VoxelGridSize.x / Param_.BlockSize.x;
    Param_.NumBlocks.y = Param_.VoxelGridSize.y / Param_.BlockSize.y;
    Param_.NumBlocks.z = Param_.VoxelGridSize.z / Param_.BlockSize.z;

    Param_.NumVoxels = Param_.VoxelGridSize.x * Param_.VoxelGridSize.y * Param_.VoxelGridSize.z;
    Param_.NumVertices = 0;
    Param_.MaxVertices = 0;

    Param_.VoxelSize = DefaultVoxelSize;
    Param_.MinPosition = make_float3(
                -Param_.VoxelSize.x * Param_.VoxelGridSize.x / 2.f,
                -Param_.VoxelSize.y * Param_.VoxelGridSize.y / 2.f,
                -Param_.VoxelSize.z * Param_.VoxelGridSize.z / 2.f
                );

    Param_.Threshold = DefaultThreshold;
    Param_.ZSlice = 0;
}

FractalObject::~FractalObject()
{
    delete[] CudaVboResources_;
}

void FractalObject::Init(ShaderProgram* Shaders, TransformMatrix* Projection, TransformMatrix* Visualization,
                         const float3* /*Vertices*/, const float3* /*Normals*/, const uint /*NumVertices*/,
                         const uint3* /*Connectivity*/, const uint NumElements,
                         const float4* /*Colors*/, const uint /*NumColors*/, const SDL_Surface* TexImage, const float2* TexCoord)
{
    MaxTriangles_ = NumElements;
    Param_.MaxVertices = MaxTriangles_ * 3;

    DisplayItem::Init(Shaders, Projection, Visualization, NULL, NULL, Param_.MaxVertices, NULL, NumElements, NULL, Param_.MaxVertices, TexImage, TexCoord);

    InitTexture();
    InitCuda();

    Param_.Print();
}

void FractalObject::Update()
{
    //LaunchTest(Param_, DevicePointers_);

    LaunchSampleVolume(Param_, DevicePointers_);

    LaunchClassifyVoxel(Param_, DevicePointers_);
    LaunchThrustScan(Param_, DevicePointers_);

    CudaCheck(cudaDeviceSynchronize());

    {
        // Set number of vertices and triangles
        uint LastElement, LastScanElement;
        CudaCheck(cudaMemcpy(&LastElement, DevicePointers_.VoxelVertices + Param_.NumVoxels - 1, sizeof(uint), cudaMemcpyDeviceToHost));
        CudaCheck(cudaMemcpy(&LastScanElement, DevicePointers_.VoxelVerticesScan + Param_.NumVoxels - 1, sizeof(uint), cudaMemcpyDeviceToHost));
        Param_.NumVertices = LastElement + LastScanElement;
        NumElements_ = min(Param_.NumVertices, Param_.MaxVertices) / 3;
    }

    CudaCheck(cudaDeviceSynchronize());

    MapBuffers();

    LaunchGenerateTriangles(Param_, DevicePointers_);
    CudaCheck(cudaDeviceSynchronize());

    CudaCheck(cudaMemcpyToArray(TexArray_, 0, 0, DevicePointers_.TexCudaTarget, TexDataSize_, cudaMemcpyDeviceToDevice));

    UnmapBuffers();
}

void FractalObject::AttachGLTexture(GLuint TextureId, uint TexDataSize)
{
    TexDataSize_ = TexDataSize;
    CudaCheck(cudaMalloc((void**)&DevicePointers_.TexCudaTarget, TexDataSize_));
    CudaCheck(cudaMemset(DevicePointers_.TexCudaTarget, 0, TexDataSize_));
    CudaCheck(cudaGraphicsGLRegisterImage(&TexResource_, TextureId, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
}

void FractalObject::MovePlane(int Distance)
{
    const int Target = Param_.ZSlice + Distance;
    if (Target >= 0 && Target < (int)Param_.VoxelGridSize.z) Param_.ZSlice = Target;
}

void FractalObject::InitTexture()
{
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &Texture_);
    glBindTexture(GL_TEXTURE_2D, Texture_);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Param_.VoxelGridSize.x, Param_.VoxelGridSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    TexDataSize_ = Param_.VoxelGridSize.x * Param_.VoxelGridSize.y * 4 * sizeof(GLubyte);
    CudaCheck(cudaMalloc((void**)&DevicePointers_.TexCudaTarget, TexDataSize_));
    CudaCheck(cudaMemset(DevicePointers_.TexCudaTarget, 0, TexDataSize_));
    CudaCheck(cudaGraphicsGLRegisterImage(&TexResource_, Texture_, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
}

void FractalObject::InitCuda()
{
    CudaVboResources_ = new cudaGraphicsResource*[NumVBO_];

    for (uint iBuffer = 0; iBuffer < NumVBO_; iBuffer++)
    {
        CudaCheck(cudaGraphicsGLRegisterBuffer(&CudaVboResources_[iBuffer], VBOs_[iBuffer], cudaGraphicsMapFlagsNone));
    }

    MapBuffers();

    CudaCheck(cudaMalloc((void**)&DevicePointers_.VoxelValues, Param_.NumVoxels * sizeof(uchar)));

    CudaCheck(cudaMalloc(&DevicePointers_.VoxelVertices, Param_.NumVoxels * sizeof(uint)));
    CudaCheck(cudaMalloc(&DevicePointers_.VoxelVerticesScan, Param_.NumVoxels * sizeof(uint)));
    CudaCheck(cudaMemset(DevicePointers_.VoxelVertices, 0, Param_.NumVoxels * sizeof(uint)));
    CudaCheck(cudaMemset(DevicePointers_.VoxelVerticesScan, 0, Param_.NumVoxels * sizeof(uint)));


    CreateTextures();

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

    if (TexResource_)
    {
        CudaCheck(cudaGraphicsMapResources(1, &TexResource_));
        CudaCheck(cudaGraphicsSubResourceGetMappedArray(&TexArray_, TexResource_, 0, 0));
    }
}

void FractalObject::UnmapBuffers()
{
    for (uint iBuffer = 0; iBuffer < NumVBO_; iBuffer++)
    {
        CudaCheck(cudaGraphicsUnmapResources(1, &CudaVboResources_[iBuffer]));
    }

    if (TexResource_)
    {
        CudaCheck(cudaGraphicsUnmapResources(1, &TexResource_));
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

/**
 * @brief Creates a single texture (with some hardcoded parameters).
 * Type is always unsigned for now
 * @param Texture
 * @param Buffer
 */
void CreateSingleTexture(cudaTextureObject_t* Texture, uint* Buffer, uint BufferSize)
{
    cudaResourceDesc ResDesc;
    memset(&ResDesc, 0, sizeof(ResDesc));
    ResDesc.resType = cudaResourceTypeLinear;
    ResDesc.res.linear.devPtr = Buffer;
    ResDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    ResDesc.res.linear.desc.x = 32; // bits per channel
    ResDesc.res.linear.sizeInBytes = BufferSize;

    cudaTextureDesc TexDesc;
    memset(&TexDesc, 0, sizeof(TexDesc));
    TexDesc.readMode = cudaReadModeElementType;

    CudaCheck(cudaCreateTextureObject(Texture, &ResDesc, &TexDesc, NULL));
}

void FractalObject::CreateTextures()
{
    CudaCheck(cudaMalloc(&DevicePointers_.EdgeTable, sizeof(EdgeTable)));
    cout << "sizeof(EdgeTable) = " << sizeof(EdgeTable) << endl;
    CudaCheck(cudaMalloc(&DevicePointers_.TriangleTable, sizeof(TriangleTable)));
    CudaCheck(cudaMalloc(&DevicePointers_.NumVertexTable, sizeof(NumVertexTable)));

    CudaCheck(cudaMemcpy(DevicePointers_.EdgeTable, EdgeTable, sizeof(EdgeTable), cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(DevicePointers_.TriangleTable, TriangleTable, sizeof(TriangleTable), cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(DevicePointers_.NumVertexTable, NumVertexTable, sizeof(NumVertexTable), cudaMemcpyHostToDevice));

    CreateSingleTexture(&DevicePointers_.EdgeTableTex, DevicePointers_.EdgeTable, sizeof(EdgeTable));
    CreateSingleTexture(&DevicePointers_.TriangleTableTex, DevicePointers_.TriangleTable, sizeof(TriangleTable));
    CreateSingleTexture(&DevicePointers_.NumVertexTableTex, DevicePointers_.NumVertexTable, sizeof(NumVertexTable));
}

void FractalObject::LoadVoxels()
{
    string Filename = "Bucky.raw";
    ifstream VoxelFile(Filename, ifstream::in | ifstream::binary);


    if (VoxelFile.is_open())
    {
        VoxelFile.seekg (0, ios::end);
        uint Length = VoxelFile.tellg();
        VoxelFile.seekg (0, ios::beg);

        Param_.NumVoxels = Length;
        char* TmpBuffer = new char[Length];
        VoxelFile.read(TmpBuffer, Length);

        CudaCheck(cudaMalloc((void**)&DevicePointers_.VoxelValues, Length));
        CudaCheck(cudaMemcpy(DevicePointers_.VoxelValues, TmpBuffer, Length, cudaMemcpyHostToDevice));

        delete[] TmpBuffer;

        Param_.VoxelGridSize = make_uint3(32, 32, 32);
        Param_.NumBlocks.x = Param_.VoxelGridSize.x / Param_.BlockSize.x;
        Param_.NumBlocks.y = Param_.VoxelGridSize.y / Param_.BlockSize.y;
        Param_.NumBlocks.z = Param_.VoxelGridSize.z / Param_.BlockSize.z;

        Param_.MinPosition = make_float3(-0.0f, -0.0f, -0.0f);
        Param_.VoxelSize = make_float3(0.05f, 0.05f, 0.05f);

        CudaCheck(cudaMalloc(&DevicePointers_.VoxelVertices, Param_.NumVoxels * sizeof(uint)));
        CudaCheck(cudaMalloc(&DevicePointers_.VoxelVerticesScan, Param_.NumVoxels * sizeof(uint)));
        CudaCheck(cudaMemset(DevicePointers_.VoxelVertices, 0, Param_.NumVoxels * sizeof(uint)));
        CudaCheck(cudaMemset(DevicePointers_.VoxelVerticesScan, 0, Param_.NumVoxels * sizeof(uint)));
    }
    else
    {
        cerr << "Unable to open \"" << Filename << "\"" << endl;
    }
}
