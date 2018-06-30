#pragma once
#ifndef HELPER_MATH_
#define HELPER_MATH

#include "cuda_runtime.h"
#include <math.h>

#define __VECTOR_FUNCTIONS_DECL__ static __inline__ __host__ __device__



inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __host__ __device__ float3 operator/(float3 a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ float3 operator*(float3 a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator*(float b, float3 a)
{
	return make_float3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ void operator+=(float3 &a, float3 b){
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

inline __host__ __device__ void operator+=(float4 &a, float4 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}

inline __host__ __device__ void operator*=(float3 &a, float b){
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ float3 make_float3(float s)
{
	return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
	return make_float3(a.x, a.y, a.z);
}

inline __host__ __device__ float3 make_float3(int3 a)
{
	return make_float3(a.x, a.y, a.z);
}

inline __host__ __device__ float4 make_float4(float3 a, float w){
	return make_float4(a.x, a.y, a.z, w);
}

inline __host__ __device__ float4 make_float4(float s)
{
	return make_float4(s, s, s, s);
}

inline __host__ __device__ float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float length(float3 v)
{
	return sqrtf(dot(v, v));
}

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

#endif