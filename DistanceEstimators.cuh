#ifndef DISTANCE_ESTIMATORS_CUH__
#define DISTANCE_ESTIMATORS_CUH__

#include "CudaMath.h"
#include "HostDeviceCode.h"

template <DEType default_dist>
__host__ __device__ float GetDistance(const float3& Position)
{
    return fmaxf(Length(Position) - 0.5f, 0.f); // Just a sphere
}

template <>
__host__ __device__ float GetDistance<Sphere>(const float3& Position)
{
    return fmaxf(Length(Position) - 0.5f, 0.f);
}

template <>
__host__ __device__ float GetDistance<TripleSphere>(const float3& Position)
{
    return fminf(fmaxf(Length(Position - make_float3(1.f, 1.f, 0.f)) - 0.5f, 0.f),
                 fminf(fmaxf(Length(Position - make_float3(1.f, -1.f, 0.f)) - 0.5f, 0.f),
                       fmaxf(Length(Position - make_float3(0.f, 0.f, 2.f)) - 0.5f, 0.f)));
}

template <>
__host__ __device__ float GetDistance<FractalTriangle>(const float3& Position)
{
    const float3 a1 = make_float3(1, 1, 1);
    const float3 a2 = make_float3(-1, -1, 1);
    const float3 a3 = make_float3(1, -1, -1);
    const float3 a4 = make_float3(-1, 1, -1);
    float3       z  = Position;
    float3       c;
    int          n = 0;
    float        dist, d;
    const int    NUM_ITERATIONS = 20;
    const float  SCALE          = 2.0f;
    while (n < NUM_ITERATIONS)
    {
        c    = a1;
        dist = Length(z - a1);
        d    = Length(z - a2);
        if (d < dist)
        {
            c    = a2;
            dist = d;
        }
        d = Length(z - a3);
        if (d < dist)
        {
            c    = a3;
            dist = d;
        }
        d = Length(z - a4);
        if (d < dist)
        {
            c    = a4;
            dist = d;
        }
        z = SCALE * z - c * (SCALE - 1.0);
        n++;
    }

    return Length(z) * pow(SCALE, float(-n));
}

#endif // DISTANCE_ESTIMATORS_CUH__
