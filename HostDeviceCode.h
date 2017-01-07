#ifndef HOSTDEVICECODE_H
#define HOSTDEVICECODE_H

#include <cstdio>
#include <cstdlib>

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

struct ArrayPointers
{
    float3* Vertices;
    float3* Normals;
    uint3* Connectivity;
    float4* Colors;
};


#endif // HOSTDEVICECODE_H
