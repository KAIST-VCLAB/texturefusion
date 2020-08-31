#pragma once

#ifndef sint
typedef signed int sint;
#endif

#ifndef uint
typedef unsigned int uint;
#endif 

#ifndef slong 
typedef signed long slong;
#endif

#ifndef ulong
typedef unsigned long ulong;
#endif

#ifndef uchar
typedef unsigned char uchar;
#endif

#ifndef schar
typedef signed char schar;
#endif

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"
#include "CUDATexPoolParams.h"
#include "VoxelUtilHashSDF.h"

#include "DepthCameraUtil.h"

//#define c_texPoolParams.m_texturePatchWidth 8

#define HALFPIXEL_CENTER
//#define INTPIXEL_CENTER

//#define DISCRETE_SAMPLING
#define BILINEAR_SAMPLING

#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif

#ifndef PINF
#define PINF __int_as_float(0x7f800000)
#endif

extern  __constant__ TexPoolParams c_texPoolParams;
extern "C" void updateConstantTexPoolParams(const TexPoolParams& texPoolParams);

__align__(16)
struct Texel {	

	uchar3 color_dummy; // previous color
	uchar3 color; //current color
	
	uchar weight;

//	uchar alpha;
	float alpha;

};

struct TexPoolData {
	
	__device__ __host__ TexPoolData() {
		d_heap = NULL;
		d_heapCounter = NULL;
		d_texPatches = NULL;
	}

	__host__
	void allocate(const TexPoolParams& params, bool dataOnGPU = true) {
		m_bIsOnGPU = dataOnGPU;
		if (m_bIsOnGPU) {

			cutilSafeCall(cudaMalloc(&d_heap, sizeof(unsigned int) * params.m_numTexturePatches));
			cutilSafeCall(cudaMalloc(&d_heapCounter, sizeof(unsigned int)));
			cutilSafeCall(cudaMalloc(&d_texPatches, sizeof(Texel) * (params.m_numTexturePatches + 1) * params.m_texturePatchSize));
			cutilSafeCall(cudaMalloc(&d_texPatchDir, sizeof(char) * (params.m_numTexturePatches + 1)));
			//std::cout << "#texture patches: " << params.m_numTexturePatches << "\n";

		}
	}

	__host__
	void updateParams(const TexPoolParams& params) {
		if (m_bIsOnGPU) {
			updateConstantTexPoolParams(params);
		}
	}

	__host__
	uint getNumTexture(uint m_numTexturePatches) {
		uint h_heapCounter;
		cudaMemcpy(&h_heapCounter, d_heapCounter, sizeof(uint), cudaMemcpyDeviceToHost);

		return m_numTexturePatches - h_heapCounter;
	}

	__host__
		void getTextureHeap(uint *h_textureHeap, uint m_numTexturePatches) {
		cudaMemcpy(h_textureHeap, d_heap, sizeof(uint) * m_numTexturePatches, cudaMemcpyDeviceToHost);
	}

	__host__
		uint getTextureHeapAddress(uint *h_heap, uint h_num_texture, uint m_numTexturePatches, uint h_texture_idx)
	{

		for (int i = m_numTexturePatches - 1; i >= m_numTexturePatches - h_num_texture; i--) {
			if (h_heap[i] == h_texture_idx)
				return i;
		}

		return m_numTexturePatches + 1;

	}

	__host__
	void free() {

		if (m_bIsOnGPU) {
			cutilSafeCall(cudaFree(d_heap));
			cutilSafeCall(cudaFree(d_heapCounter));
			cutilSafeCall(cudaFree(d_texPatches));
		}

		d_heap = NULL;
		d_heapCounter = NULL;
		d_texPatches = NULL;

	}

//#define __CUDACC__
#ifdef __CUDACC__

	__device__
		float frac(float val) const {
		return (val - floorf(val));
	}

	__device__
		float3 frac(const float3& val) const {
		return make_float3(frac(val.x), frac(val.y), frac(val.z));
	}

	__device__
		uint linearizeTexelIndex(uint texture_ind, uint2 uv) const {
		return texture_ind * c_texPoolParams.m_texturePatchWidth* c_texPoolParams.m_texturePatchWidth + uv.y * c_texPoolParams.m_texturePatchWidth + uv.x;
	}

	__device__
		uint2 delinearizeTexelIndex(uint idx) const {
		uint x = idx % c_texPoolParams.m_texturePatchWidth;
		uint y = idx / c_texPoolParams.m_texturePatchWidth;
		return make_uint2(x, y);
	}

	__device__
		float3 getWorldTexDir(const char &dir) const {

		if (dir == 0) {
			return make_float3(1.f, 0.f, 0.f);
		}
		else if (dir == 1) {
			return make_float3(-1.f, 0.f, 0.f);
		}
		else if (dir == 2) {
			return make_float3(0.f, 1.f, 0.f);

		}
		else if (dir == 3) {
			return make_float3(0.f, -1.f, 0.f);
		}
		else if (dir == 4) {
			return make_float3(0.f, 0.f, 1.f);
		}
		else if (dir == 5) {
			return make_float3(0.f, 0.f, -1.f);
		}
		return make_float3(0.f, 0.f, 0.f);
	}

	__device__
		float3 uvToVirtualLocalVoxelPos(const uint2 &uvv, char dir) const {

#ifdef HALFPIXEL_CENTER

		float x = 0.f;
		float y = 0.f;
		float z = 0.f;

		float2 uv = make_float2(uvv);

		if (dir == 0) {
			x = 0.f;
			y = (uv.x + 0.5f) / (float)c_texPoolParams.m_texturePatchWidth;
			z = (uv.y + 0.5f) / (float)c_texPoolParams.m_texturePatchWidth;
		}
		else if (dir == 1) {
			x = 1.f;
			y = (uv.x + 0.5f) / (float)c_texPoolParams.m_texturePatchWidth;
			z = (uv.y + 0.5f) / (float)c_texPoolParams.m_texturePatchWidth;
		}
		else if (dir == 2) {
			x = (uv.x + 0.5f) / (float)c_texPoolParams.m_texturePatchWidth;
			y = 0.f;
			z = (uv.y + 0.5f) / (float)c_texPoolParams.m_texturePatchWidth;
		}
		else if (dir == 3) {
			x = (uv.x + 0.5f) / (float)c_texPoolParams.m_texturePatchWidth;
			y = 1.f;
			z = (uv.y + 0.5f) / (float)c_texPoolParams.m_texturePatchWidth;

		}
		else if (dir == 4) {
			x = (uv.x + 0.5f) / (float)c_texPoolParams.m_texturePatchWidth;
			y = (uv.y + 0.5f) / (float)c_texPoolParams.m_texturePatchWidth;
			z = 0.f;
		}
		else if (dir == 5) {
			x = (uv.x + 0.5f) / (float)c_texPoolParams.m_texturePatchWidth;
			y = (uv.y + 0.5f) / (float)c_texPoolParams.m_texturePatchWidth;
			z = 1.f;
		}

		return make_float3(x,y,z);

#endif

#ifdef INTPIXEL_CENTER

		float x = 0.f;
		float y = 0.f;
		float z = 0.f;

		float2 uv = make_float2(uvv);
		uv = uv / (float)(c_texPoolParams.m_texturePatchWidth - 1);
		if (uv.x > 1.f)
			uv.x = 0.999f;
		if (uv.y > 1.f)
			uv.x = 0.999f;
		if (uv.x < 0)
			uv.x = 0.001f;
		if (uv.y < 0)
			uv.y = 0.001f;
		if (dir == 0) {
			x = 0.f;
			y =uv.x;
			z =uv.y;
		}
		else if (dir == 1) {
			x = 1.f;
			y= uv.x;
			z= uv.y;
		}
		else if (dir == 2) {
			x =uv.x;
			y = 0.f;
			z =uv.y;
		}
		else if (dir == 3) {
			x =uv.x;
			y = 1.f;
			z =uv.y;

		}
		else if (dir == 4) {
			x =uv.x;
			y =uv.y;
			z = 0.f;
		}
		else if (dir == 5) {
			x =uv.x;
			y= uv.y;
			z = 1.f;
		}

		return make_float3(x, y, z);

#endif

	}

	__device__
		float3 uvToVirtualLocalVoxelPosPad(const uint2 &uvv, char dir) const {

		float x = 0.f;
		float y = 0.f;
		float z = 0.f;

		float2 uv = make_float2(uvv);

		if (dir == 0) {
			x = 0.f;
			y = (uv.x ) / (float)(c_texPoolParams.m_texturePatchWidth-1);
			z = (uv.y ) / (float)(c_texPoolParams.m_texturePatchWidth-1);
		}
		else if (dir == 1) {
			x = 1.f;
			y = (uv.x ) / (float)(c_texPoolParams.m_texturePatchWidth-1);
			z = (uv.y ) / (float)(c_texPoolParams.m_texturePatchWidth-1);
		}
		else if (dir == 2) {
			x = (uv.x ) / (float)(c_texPoolParams.m_texturePatchWidth-1);
			y = 0.f;
			z = (uv.y ) / (float)(c_texPoolParams.m_texturePatchWidth-1);
		}
		else if (dir == 3) {
			x = (uv.x ) / (float)(c_texPoolParams.m_texturePatchWidth-1);
			y = 1.f;
			z = (uv.y ) / (float)(c_texPoolParams.m_texturePatchWidth-1);

		}
		else if (dir == 4) {
			x = (uv.x ) / (float)(c_texPoolParams.m_texturePatchWidth-1);
			y = (uv.y ) / (float)(c_texPoolParams.m_texturePatchWidth-1);
			z = 0.f;
		}
		else if (dir == 5) {
			x = (uv.x ) / (float)(c_texPoolParams.m_texturePatchWidth-1);
			y = (uv.y ) / (float)(c_texPoolParams.m_texturePatchWidth-1);
			z = 1.f;
		}

		return make_float3(x, y, z);

	}
	
	__device__
	char getTexDir(const int &ind)
	{
		return d_texPatchDir[ind];
	}

	/////////////////////////////////////////////////////////////////////////
	// manage tile heap sturcture
	/////////////////////////////////////////////////////////////////////////

	__device__
		uint consumeHeap() {
		uint addr = atomicSub(&d_heapCounter[0], 1);
		return d_heap[addr];
	}

	__device__
		uint consumeHeap(int count) {
		uint addr = atomicSub(&d_heapCounter[0], count);
		return addr - count + 1;
	}

	__device__
		void appendHeap(uint ptr) {
		uint addr = atomicAdd(&d_heapCounter[0], 1);
		d_heap[addr + 1] = ptr;
	}

	__device__
		uint appendHeapV(int count) {
		uint addr = atomicAdd(&d_heapCounter[0], count);
		return addr + 1;
	}

	/////////////////////////////////////////////////////////////////////////
	// Find texel
	/////////////////////////////////////////////////////////////////////////

	__device__
		Texel getTexel(uint texture_ind, uint2 uv) const {

		Texel t;
		t = d_texPatches[texture_ind * c_texPoolParams.m_texturePatchWidth * c_texPoolParams.m_texturePatchWidth + uv.y * c_texPoolParams.m_texturePatchWidth + uv.x];
		return t;
	}

	__device__
		Texel getTexelDiscrete(uint texture_ind, float2 uv) const {

		Texel t;
		uint2 uvu = make_uint2(uv);
		t = d_texPatches[texture_ind * c_texPoolParams.m_texturePatchWidth * c_texPoolParams.m_texturePatchWidth + uvu.y * c_texPoolParams.m_texturePatchWidth + uvu.x];
		return t;
	}

	__device__
		Texel getTexelBilinear(uint texture_ind, float2 uv) const {

		Texel ld;
		Texel lu;
		Texel rd;
		Texel ru;

#ifdef HALFPIXEL_CENTER
		uv -= 0.5;
#endif

		if (uv.x >= (float)c_texPoolParams.m_texturePatchWidth - 1)
			uv.x = (float)c_texPoolParams.m_texturePatchWidth - 1.000001;
		if (uv.y >= (float)c_texPoolParams.m_texturePatchWidth - 1)
			uv.y = (float)c_texPoolParams.m_texturePatchWidth - 1.000001;
		if (uv.x <= 0)
			uv.x = 0.000001;
		if (uv.y <= 0)
			uv.y = 0.000001;

		uint2 uvu = make_uint2(uv.x, uv.y);
		float2 weight = make_float2(uv.x - floorf(uv.x), uv.y - floorf(uv.y));
		ld = d_texPatches[texture_ind * c_texPoolParams.m_texturePatchWidth * c_texPoolParams.m_texturePatchWidth + uvu.y * c_texPoolParams.m_texturePatchWidth + uvu.x];
		rd = d_texPatches[texture_ind * c_texPoolParams.m_texturePatchWidth * c_texPoolParams.m_texturePatchWidth + uvu.y * c_texPoolParams.m_texturePatchWidth + uvu.x + 1];
		lu = d_texPatches[texture_ind * c_texPoolParams.m_texturePatchWidth * c_texPoolParams.m_texturePatchWidth + (uvu.y + 1) * c_texPoolParams.m_texturePatchWidth + uvu.x];
		ru = d_texPatches[texture_ind * c_texPoolParams.m_texturePatchWidth * c_texPoolParams.m_texturePatchWidth + (uvu.y + 1) * c_texPoolParams.m_texturePatchWidth + uvu.x + 1];

		Texel t;
		t.color.x = 0;
		t.color.y = 0;
		t.color.z = 0;

		t.color_dummy.x = 0;
		t.color_dummy.y = 0;
		t.color_dummy.z = 0;
		t.alpha = 0.;
		t.weight = 0;

		float val = 0.f;
		float3 colordummy;
		float3 color, colortmp;
		float weightf, alpha;
		weightf = 0.f;
		alpha = 0.f;
		color = make_float3(0.f);
		colordummy = make_float3(0.f);
		colortmp = make_float3(0.f);


		/*t.color_dummy = weight.x * weight.y* ru.color_dummy + weight.x * (1 - weight.y)* rd.color_dummy +
		(1 - weight.x) * weight.y* lu.color_dummy + (1 - weight.x) * (1 - weight.y)* ld.color_dummy;
		t.color = weight.x * weight.y* ru.color + weight.x * (1 - weight.y)* rd.color +
		(1 - weight.x) * weight.y* lu.color + (1 - weight.x) * (1 - weight.y)* ld.color;
		*/


		if (ld.weight >= 0.f) {
			colortmp = make_float3(ld.color_dummy);
			colordummy += (1 - weight.x) * (1 - weight.y)*colortmp;
			colortmp = make_float3(ld.color);
			color += (1 - weight.x) * (1 - weight.y) * colortmp;
			weightf += (1 - weight.x) * (1 - weight.y) * ld.weight;
			alpha += (1 - weight.x) * (1 - weight.y) * ld.alpha;
			val += (1 - weight.x) * (1 - weight.y);
		}
		if (rd.weight >= 0.f) {
			colortmp = make_float3(rd.color_dummy);
			colordummy += weight.x * (1 - weight.y)*colortmp;
			colortmp = make_float3(rd.color);
			color += weight.x * (1 - weight.y) * colortmp;
			weightf += (1 - weight.y) * weight.x* rd.weight;
			alpha += (1 - weight.y) * weight.x* rd.alpha;
			val += weight.x * (1 - weight.y);
		}
		if (lu.weight >= 0.f) {
			colortmp = make_float3(lu.color_dummy);
			colordummy += (1 - weight.x) * weight.y*colortmp;
			colortmp = make_float3(lu.color);
			color += (1 - weight.x) * weight.y * colortmp;
			weightf += (1 - weight.x) * weight.y * lu.weight;
			alpha += (1 - weight.x) * weight.y * lu.alpha;
			val += (1 - weight.x) * weight.y;
		}
		if (ru.weight >= 0.f) {
			colortmp = make_float3(ru.color_dummy);
			colordummy += weight.x * weight.y*colortmp;
			colortmp = make_float3(ru.color);
			color += weight.x * weight.y * colortmp;
			weightf += weight.x * weight.y * ru.weight;
			alpha += weight.x * weight.y * ru.alpha;
			val += weight.x * weight.y;
		}

		color /= val;
		colordummy /= val;
		t.color = make_uchar3(color.x, color.y, color.z);
		t.color_dummy = make_uchar3(colordummy.x, colordummy.y, colordummy.z);
		t.alpha = alpha / val;
		t.weight = weightf / val;

		return t;
	}
	
	__device__
		Texel getTexel(const float3& weight, const int texture_ind) const {

		Texel texel;

		if (texture_ind > 0) {
			int texDir = d_texPatchDir[texture_ind];

#ifdef INTPIXEL_CENTER
			float2 uv;
			uchar3 dullcolor;

			if (texDir == 0 || texDir == 1) {
				uv.x = weight.y * (float)(c_texPoolParams.m_texturePatchWidth - 1);
				uv.y = weight.z * (float)(c_texPoolParams.m_texturePatchWidth - 1);

				if (texDir == 0)
					dullcolor = make_uchar3(255, 255, 0);
				else dullcolor = make_uchar3(0, 0, 255);
			}

			if (texDir == 2 || texDir == 3) {
				uv.x = weight.x * (float)(c_texPoolParams.m_texturePatchWidth - 1);
				uv.y = weight.z * (float)(c_texPoolParams.m_texturePatchWidth - 1);
				if (texDir == 2)
					dullcolor = make_uchar3(255, 0, 255);
				else dullcolor = make_uchar3(0, 255, 0);
			}

			if (texDir == 4 || texDir == 5) {
				uv.x = weight.x * (float)(c_texPoolParams.m_texturePatchWidth - 1);
				uv.y = weight.y * (float)(c_texPoolParams.m_texturePatchWidth - 1);
				if (texDir == 4)
					dullcolor = make_uchar3(0, 255, 255);
				else dullcolor = make_uchar3(255, 0, 0);
			}

#ifdef BILINEAR_SAMPLING
			texel = getTexelBilinear(texture_ind, uv);
#endif

#ifdef DISCRETE_SAMPLING
			texel = getTexelDiscrete(texture_ind, uv + 0.5);
#endif

#endif

#ifdef HALFPIXEL_CENTER
			float2 uv;
			uchar3 dullcolor;

			if (texDir == 0 || texDir == 1) {
				uv.x = weight.y * (float)c_texPoolParams.m_texturePatchWidth;
				uv.y = weight.z * (float)c_texPoolParams.m_texturePatchWidth;

				if (texDir == 0)
					dullcolor = make_uchar3(255, 255, 0);
				else dullcolor = make_uchar3(0, 0, 255);
			}

			if (texDir == 2 || texDir == 3) {
				uv.x = weight.x * (float)c_texPoolParams.m_texturePatchWidth;
				uv.y = weight.z * (float)c_texPoolParams.m_texturePatchWidth;
				if (texDir == 2)
					dullcolor = make_uchar3(255, 0, 255);
				else dullcolor = make_uchar3(0, 255, 0);
			}

			if (texDir == 4 || texDir == 5) {
				uv.x = weight.x * (float)c_texPoolParams.m_texturePatchWidth;
				uv.y = weight.y * (float)c_texPoolParams.m_texturePatchWidth;
				if (texDir == 4)
					dullcolor = make_uchar3(0, 255, 255);
				else dullcolor = make_uchar3(255, 0, 0);
			}

#ifdef BILINEAR_SAMPLING
			texel = getTexelBilinear(texture_ind, uv);
#endif

#ifdef DISCRETE_SAMPLING
			texel = getTexelDiscrete(texture_ind, uv);
#endif
#endif

		}
		else {
			texel.color = make_uchar3(MINF, MINF, MINF);
			texel.color_dummy = make_uchar3(MINF, MINF, MINF);
			texel.weight = 0.f;
			texel.alpha = -1;
		}

		return texel;
	}


	__device__
		Texel getTexelPrev(const float3& weight, const int texture_ind) const {

		Texel texel;

		if (texture_ind > 0) {
			int texDir = d_texPatchDir[texture_ind];

#ifdef INTPIXEL_CENTER
			float2 uv;
			uchar3 dullcolor;

			if (texDir == 0 || texDir == 1) {
				uv.x = weight.y * (c_texPoolParams.m_texturePatchWidth - 1);
				uv.y = weight.z * (c_texPoolParams.m_texturePatchWidth - 1);

				if (texDir == 0)
					dullcolor = make_uchar3(255, 255, 0);
				else dullcolor = make_uchar3(0, 0, 255);
			}

			if (texDir == 2 || texDir == 3) {
				uv.x = weight.x * (c_texPoolParams.m_texturePatchWidth - 1);
				uv.y = weight.z * (c_texPoolParams.m_texturePatchWidth - 1);
				if (texDir == 2)
					dullcolor = make_uchar3(255, 0, 255);
				else dullcolor = make_uchar3(0, 255, 0);
			}

			if (texDir == 4 || texDir == 5) {
				uv.x = weight.x * (c_texPoolParams.m_texturePatchWidth - 1);
				uv.y = weight.y * (c_texPoolParams.m_texturePatchWidth - 1);
				if (texDir == 4)
					dullcolor = make_uchar3(0, 255, 255);
				else dullcolor = make_uchar3(255, 0, 0);
			}

			texel = getTexelDiscrete(texture_ind, uv + 0.5);

#endif


#ifdef HALFPIXEL_CENTER
			float2 uv;
			uchar3 dullcolor;

			if (texDir == 0 || texDir == 1) {
				uv.x = weight.y * c_texPoolParams.m_texturePatchWidth;
				uv.y = weight.z * c_texPoolParams.m_texturePatchWidth;

				if (texDir == 0)
					dullcolor = make_uchar3(255, 255, 0);
				else dullcolor = make_uchar3(0, 0, 255);
			}

			if (texDir == 2 || texDir == 3) {
				uv.x = weight.x * c_texPoolParams.m_texturePatchWidth;
				uv.y = weight.z * c_texPoolParams.m_texturePatchWidth;
				if (texDir == 2)
					dullcolor = make_uchar3(255, 0, 255);
				else dullcolor = make_uchar3(0, 255, 0);
			}

			if (texDir == 4 || texDir == 5) {
				uv.x = weight.x * c_texPoolParams.m_texturePatchWidth;
				uv.y = weight.y * c_texPoolParams.m_texturePatchWidth;
				if (texDir == 4)
					dullcolor = make_uchar3(0, 255, 255);
				else dullcolor = make_uchar3(255, 0, 0);
			}


			texel = getTexelDiscrete(texture_ind, uv);
#endif




		}
		else {
			texel.color = make_uchar3(MINF, MINF, MINF);
			texel.color_dummy = make_uchar3(MINF, MINF, MINF);
			texel.weight = 0.f;
			texel.alpha = -1;
		}
		return texel;
	}

	__device__
		Texel getTexel(const float3& worldPos, HashData hashData) const {
		
		//compute offset
		const float oSet = c_hashParams.m_virtualVoxelSize; 
		const float3 posDual = worldPos - make_float3(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f);

		//find texel
		Voxel v = hashData.getVoxel(posDual);
		float3 weight = frac(hashData.worldToVirtualVoxelPosFloat(worldPos)); // a - floor (a)
				
		return getTexel(weight, v.texind);
	}
	
	__device__
		Texel getTexelPrev(const float3& worldPos, HashData hashData) const {

		//compute offset
		const float oSet = c_hashParams.m_virtualVoxelSize;
		const float3 posDual = worldPos - make_float3(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f); // relative position of position
		
		//find texel
		Voxel v = hashData.getVoxel(posDual);
		float3 weight = frac(hashData.worldToVirtualVoxelPosFloat(worldPos)); // a - floor (a)
																			  
		return getTexelPrev(weight, v.prev_texind);

	}

	__device__
		void deleteTexel(Texel& t) const {
		t.color = make_uchar3(MINF, MINF, MINF);
		t.color_dummy = make_uchar3(MINF, MINF, MINF);
		t.alpha = -10.f;
		t.weight = 0.f;
	}

	__device__
	void deleteTexel(uint id) {
		deleteTexel(d_texPatches[id]);
	}

#endif
	
	uint *d_heap;
	uint *d_heapCounter;
	Texel *d_texPatches;
	char *d_texPatchDir;
	bool m_bIsOnGPU;
	int d_textureWidth;

};