#ifndef HOSTDEVICECODE_H
#define HOSTDEVICECODE_H

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Common.h"

#define CudaCheck(call) { CudaCallWithCheck((call), __FILE__, __LINE__, #call); }

inline void CudaCallWithCheck(cudaError_t ReturnCode, const char* Filename, int LineNumber, const char* LineCode, bool bDoAbort = true)
{
    if (ReturnCode != cudaSuccess)
    {
        fprintf(stderr, "[ERROR] %s (%s: %d)\n        \"%s\"\n", cudaGetErrorString(ReturnCode), Filename, LineNumber, LineCode);
        if (bDoAbort) exit(ReturnCode);
    }
}

enum DEType
{
    Sphere,
    TripleSphere,
    FractalTriangle
};

struct ArrayPointers
{
    float3* Vertices;
    float3* Normals;
    uint3* Connectivity;
    float4* Colors;

    uchar* VoxelValues;

    uint* VoxelVertices;
    uint* VoxelVerticesScan;

    uint* EdgeTable;
    uint* TriangleTable;
    uint* NumVertexTable;

    cudaTextureObject_t EdgeTableTex;
    cudaTextureObject_t TriangleTableTex;
    cudaTextureObject_t NumVertexTableTex;

    uint* TexCudaTarget;

    void Nullify()
    {
        Vertices = NULL;
        Normals = NULL;
        Connectivity = NULL;
        Colors = NULL;

        VoxelValues = NULL;

        VoxelVertices = NULL;
        VoxelVerticesScan = NULL;

        EdgeTable = NULL;
        TriangleTable = NULL;
        NumVertexTable = NULL;

        TexCudaTarget = NULL;
    }
};

struct KernelParameters
{
    uint3 BlockSize;
    uint3 NumBlocks;

    uint NumVoxels;
    uint3 VoxelGridSize;
    float3 VoxelSize;
    float3 MinPosition;

    uint NumVertices;
    uint MaxVertices;

    float Threshold;

    uint ZSlice;

    void Print()
    {
        fprintf(stdout,
                "BlockSize: %d %d %d\n"
                "NumBlocks: %d %d %d\n"
                "NumVoxels: %d\n"
                "VoxelGridSize: %d %d %d\n"
                "VoxelSize: %.2f %.2f %.2f\n"
                "MinPosition: %.2f %.2f %.2f\n"
                "NumVertices: %d\n"
                "MaxVertices: %d\n"
                "Threshold: %.2f\n"
                "ZSlice: %d\n",
                BlockSize.x, BlockSize.y, BlockSize.z,
                NumBlocks.x, NumBlocks.y, NumBlocks.z,
                NumVoxels,
                VoxelGridSize.x, VoxelGridSize.y, VoxelGridSize.z,
                VoxelSize.x, VoxelSize.y, VoxelSize.z,
                MinPosition.x, MinPosition.y, MinPosition.z,
                NumVertices,
                MaxVertices,
                Threshold,
                ZSlice);
    }
};

struct RayMarchingParam
{
    uint3 NumBlocks; //!< Number of blocks (3D) in kernel launch
    uint3 BlockSize; //!< Number of threads (3D) in each launched block

    uint* TexCuda; //!< The texture that will contain the result
    uint2 Size; //!< Size in pixels of the displayed texture

    // Camera info
    float3 CameraPos;
    float3 CameraDir;
    float3 CameraUp;

    // Perspective info
};




#endif // HOSTDEVICECODE_H
