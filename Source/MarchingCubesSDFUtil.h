#pragma once

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "Tables.h"

extern  __constant__ TexPoolParams c_texPoolParams;
extern "C" void updateConstantTexPoolParams(const TexPoolParams& texPoolParams);

struct MarchingCubesParams {
	bool m_boxEnabled;
	float3 m_minCorner;

	unsigned int m_maxNumTriangles;
	float3 m_maxCorner;

	unsigned int m_sdfBlockSize;
	unsigned int m_hashNumBuckets;
	unsigned int m_hashBucketSize;
	float m_threshMarchingCubes;
	float m_threshMarchingCubes2;
	float3 dummy;
};



struct MarchingCubesData {

	///////////////
	// Host part //
	///////////////

	struct Vertex
	{
		float3 p;
		float3 c;

		float3 uv;
	};

	struct Triangle
	{
		Vertex v0;
		Vertex v1;
		Vertex v2;
	};

	__device__ __host__
		MarchingCubesData() {
		d_params = NULL;

		d_numOccupiedBlocks = NULL;
		d_occupiedBlocks = NULL;

		d_triangles = NULL;
		d_numTriangles = NULL;
		m_bIsOnGPU = false;
	}

	__host__
		void allocate(const MarchingCubesParams& params, bool dataOnGPU = true) {

		//TODO max blocks 
		uint maxBlocks = params.m_hashNumBuckets*params.m_hashBucketSize;

		m_bIsOnGPU = dataOnGPU;
		if (m_bIsOnGPU) {
			cutilSafeCall(cudaMalloc(&d_params, sizeof(MarchingCubesParams)));

			cutilSafeCall(cudaMalloc(&d_numOccupiedBlocks, sizeof(uint)));
			cutilSafeCall(cudaMalloc(&d_occupiedBlocks, sizeof(uint)*maxBlocks));

			cutilSafeCall(cudaMalloc(&d_triangles, sizeof(Triangle)* params.m_maxNumTriangles));
			cutilSafeCall(cudaMalloc(&d_numTriangles, sizeof(uint)));
		}
		else {
			d_params = new MarchingCubesParams;

			//don't really need those on the CPU...
			//d_numOccupiedBlocks = new uint;
			//d_occupiedBlocks = new uint[maxBlocks];

			d_triangles = new Triangle[params.m_maxNumTriangles];
			d_numTriangles = new uint;
		}
	}

	__host__
		void updateParams(const MarchingCubesParams& params) {
		if (m_bIsOnGPU) {
			cutilSafeCall(cudaMemcpy(d_params, &params, sizeof(MarchingCubesParams), cudaMemcpyHostToDevice));
		}
		else {
			*d_params = params;
		}
	}

	__host__
		void free() {
		if (m_bIsOnGPU) {
			cutilSafeCall(cudaFree(d_params));

			cutilSafeCall(cudaFree(d_numOccupiedBlocks));
			cutilSafeCall(cudaFree(d_occupiedBlocks));

			cutilSafeCall(cudaFree(d_triangles));
			cutilSafeCall(cudaFree(d_numTriangles));
		}
		else {
			if (d_params) delete d_params;

			if (d_numOccupiedBlocks) delete d_numOccupiedBlocks;
			if (d_occupiedBlocks) delete[] d_occupiedBlocks;

			if (d_triangles) delete[] d_triangles;
			if (d_numTriangles) delete d_numTriangles;
		}

		d_params = NULL;

		d_numOccupiedBlocks = NULL;
		d_occupiedBlocks = NULL;

		d_triangles = NULL;
		d_numTriangles = NULL;
	}

	//note: does not copy occupiedBlocks and occupiedVoxels
	__host__
		MarchingCubesData copyToCPU() const {
		MarchingCubesParams params;
		cutilSafeCall(cudaMemcpy(&params, d_params, sizeof(MarchingCubesParams), cudaMemcpyDeviceToHost));

		MarchingCubesData data;
		data.allocate(params, false);	// allocate the data on the CPU
		cutilSafeCall(cudaMemcpy(data.d_params, d_params, sizeof(MarchingCubesParams), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(data.d_numTriangles, d_numTriangles, sizeof(uint), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(data.d_triangles, d_triangles, sizeof(Triangle) * (params.m_maxNumTriangles), cudaMemcpyDeviceToHost));
		return data;	//TODO MATTHIAS look at this (i.e,. when does memory get destroyed ; if it's in the destructor it would kill everything here 
	}

	__host__ unsigned int getNumOccupiedBlocks() const {
		unsigned int res = 0;
		cutilSafeCall(cudaMemcpy(&res, d_numOccupiedBlocks, sizeof(uint), cudaMemcpyDeviceToHost));
		return res;
	}

	/////////////////
	// Device part //
	/////////////////
#ifdef __CUDACC__

//#ifdef __CUDACC__
//#else
//	__host__
//#endif
	__device__
		float frac_t(float val) const {
		return (val - floorf(val));
	}

	__device__
		float3 frac_t(const float3& val) const {
		return make_float3(frac_t(val.x), frac_t(val.y), frac_t(val.z));
	}

	__device__
		Vertex vertexInterp(const HashData& hashData, const TexPoolData& texPoolData, float P, int texind, char texDir, const float3& voxel_pos, const float& voxel_size, float isolevel, const float3& p1, const float3& p2, float d1, float d2, const uchar3& c1, const uchar3& c2) const
	{
		//const uchar3 c = v.color;
		Vertex r1; r1.p = p1; r1.c = make_float3(c1.x, c1.y, c1.z) / 255.f;
		Vertex r2; r2.p = p2; r2.c = make_float3(c2.x, c2.y, c2.z) / 255.f;

		Voxel v1 = hashData.getVoxel(r1.p);
		Voxel v2 = hashData.getVoxel(r2.p);
		r1.c = make_float3(v1.color.x, v1.color.y, v1.color.z) / 255.f;
		r2.c = make_float3(v2.color.x, v2.color.y, v2.color.z) / 255.f;

		if (texind >= 0) {
			int texDir1 = texPoolData.d_texPatchDir[texind];
			float3 weight1 = (hashData.worldToVirtualVoxelPosFloat(r1.p) - voxel_pos);
			r1.uv.z = float(texind + 1e-3);

			if (texDir1 == 0 || texDir1 == 1) {
				r1.uv.x = weight1.y * (float)(c_texPoolParams.m_texturePatchWidth);
				r1.uv.y = weight1.z * (float)(c_texPoolParams.m_texturePatchWidth);
			}
			if (texDir1 == 2 || texDir1 == 3) {
				r1.uv.x = weight1.x * (float)(c_texPoolParams.m_texturePatchWidth);
				r1.uv.y = weight1.z * (float)(c_texPoolParams.m_texturePatchWidth);
			}

			if (texDir1 == 4 || texDir1 == 5) {
				r1.uv.x = weight1.x * (float)(c_texPoolParams.m_texturePatchWidth);
				r1.uv.y = weight1.y * (float)(c_texPoolParams.m_texturePatchWidth);
			}

			if (r1.uv.x < 0 || r1.uv.x >= c_texPoolParams.m_texturePatchWidth)
			{
				r1.uv.x = min(c_texPoolParams.m_texturePatchWidth - 0.00001f, r1.uv.x);
				r1.uv.x = max(0.00001f, r1.uv.x);
			}
			if (r1.uv.y < 0 || r1.uv.y >= c_texPoolParams.m_texturePatchWidth)
			{
				r1.uv.y = min(c_texPoolParams.m_texturePatchWidth - 0.00001f, r1.uv.y);
				r1.uv.y = max(0.00001f, r1.uv.y);
			}
			//Texel t1 = texPoolData.getTexel(texind, make_uint2(r1.uv.x, r1.uv.y));
		}
		else {
			r1.uv = make_float3(0.0f);
		}

		if (texind >= 0) {
			int texDir2 = texPoolData.d_texPatchDir[texind];
			float3 weight2 = hashData.worldToVirtualVoxelPosFloat(r2.p) - voxel_pos;
			r2.uv.z = float(texind + 1e-3);

			if (texDir2 == 0 || texDir2 == 1) {
				r2.uv.x = weight2.y * (float)(c_texPoolParams.m_texturePatchWidth);
				r2.uv.y = weight2.z * (float)(c_texPoolParams.m_texturePatchWidth);
			}
			if (texDir2 == 2 || texDir2 == 3) {
				r2.uv.x = weight2.x * (float)(c_texPoolParams.m_texturePatchWidth);
				r2.uv.y = weight2.z * (float)(c_texPoolParams.m_texturePatchWidth);
			}
			if (texDir2 == 4 || texDir2 == 5) {
				r2.uv.x = weight2.x * (float)(c_texPoolParams.m_texturePatchWidth);
				r2.uv.y = weight2.y * (float)(c_texPoolParams.m_texturePatchWidth);
			}
			if (r2.uv.x < 0 || r2.uv.x >= c_texPoolParams.m_texturePatchWidth)
			{
				r2.uv.x = min(c_texPoolParams.m_texturePatchWidth - 0.00001f, r2.uv.x);
				r2.uv.x = max(0.00001f, r2.uv.x);
			}
			if (r2.uv.y < 0 || r2.uv.y >= c_texPoolParams.m_texturePatchWidth)
			{
				r2.uv.y = min(c_texPoolParams.m_texturePatchWidth - 0.00001f, r2.uv.y);
				r2.uv.y = max(0.00001f, r2.uv.y);
			}

			//Texel t2 = texPoolData.getTexel(texind, make_uint2(r2.uv.x, r2.uv.y));
		}
		else {
			r2.uv = make_float3(0.0f);
		}

		if (abs(isolevel - d1) < 0.00001f)		return r1;
		if (abs(isolevel - d2) < 0.00001f)		return r2;
		if (abs(d1 - d2) < 0.00001f)			return r1;

		float mu = (isolevel - d1) / (d2 - d1);

		Vertex res;
		res.p.x = p1.x + mu * (p2.x - p1.x); // Positions
		res.p.y = p1.y + mu * (p2.y - p1.y);
		res.p.z = p1.z + mu * (p2.z - p1.z);

		res.c.x = (float)(c1.x + mu * (float)(c2.x - c1.x)) / 255.f; // Color
		res.c.y = (float)(c1.y + mu * (float)(c2.y - c1.y)) / 255.f;
		res.c.z = (float)(c1.z + mu * (float)(c2.z - c1.z)) / 255.f;

		if (texind >= 0) {
			int texDir3 = texPoolData.d_texPatchDir[texind];
			float3 weight = hashData.worldToVirtualVoxelPosFloat(res.p) - voxel_pos;
			res.uv.z = float(texind + 1e-3);

			if (texDir3 == 0 || texDir3 == 1) {
				res.uv.x = weight.y * (float)(c_texPoolParams.m_texturePatchWidth);
				res.uv.y = weight.z * (float)(c_texPoolParams.m_texturePatchWidth);
			}
			if (texDir3 == 2 || texDir3 == 3) {
				res.uv.x = weight.x * (float)(c_texPoolParams.m_texturePatchWidth);
				res.uv.y = weight.z * (float)(c_texPoolParams.m_texturePatchWidth);
			}
			if (texDir3 == 4 || texDir3 == 5) {
				res.uv.x = weight.x * (float)(c_texPoolParams.m_texturePatchWidth);
				res.uv.y = weight.y * (float)(c_texPoolParams.m_texturePatchWidth);
			}
			if (res.uv.x < 0 || res.uv.x >= c_texPoolParams.m_texturePatchWidth)
			{
				res.uv.x = min(c_texPoolParams.m_texturePatchWidth - 0.00001f, res.uv.x);
				res.uv.x = max(0.00001f, res.uv.x);
			}
			if (res.uv.y < 0 || res.uv.y >= c_texPoolParams.m_texturePatchWidth)
			{
				res.uv.y = min(c_texPoolParams.m_texturePatchWidth - 0.00001f, res.uv.y);
				res.uv.y = max(0.00001f, res.uv.y);
			}

			//Texel t = texPoolData.getTexel(texind, make_uint2(res.uv.x, res.uv.y));
		}
		else {
			res.uv = make_float3(0.0f);
		}

		return res;
	}
	__device__
		void extractIsoSurfaceAtPosition(const float3& worldPos, const HashData& hashData, const RayCastData& rayCastData, const TexPoolData& texPoolData)
	{
		const HashParams& hashParams = c_hashParams;
		const MarchingCubesParams& params = *d_params;

		if (params.m_boxEnabled == 1) {
			if (!isInBoxAA(params.m_minCorner, params.m_maxCorner, worldPos)) return;
		}

		const float isolevel = 0.0f;
		uint cubeindex = 0;
		const float voxelSize = hashParams.m_virtualVoxelSize;

		Vertex vertlist[12];

		const float P = (float)hashParams.m_virtualVoxelSize + 0.0001f;
		const float M = 0.0001f;

		float3 p000 = worldPos + make_float3(M, M, M); float dist000; uchar3 color000; bool valid000 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p000, dist000, color000);
		float3 p100 = worldPos + make_float3(P, M, M); float dist100; uchar3 color100; bool valid100 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p100, dist100, color100);
		float3 p010 = worldPos + make_float3(M, P, M); float dist010; uchar3 color010; bool valid010 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p010, dist010, color010);
		float3 p001 = worldPos + make_float3(M, M, P); float dist001; uchar3 color001; bool valid001 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p001, dist001, color001);
		float3 p110 = worldPos + make_float3(P, P, M); float dist110; uchar3 color110; bool valid110 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p110, dist110, color110);
		float3 p011 = worldPos + make_float3(M, P, P); float dist011; uchar3 color011; bool valid011 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p011, dist011, color011);
		float3 p101 = worldPos + make_float3(P, M, P); float dist101; uchar3 color101; bool valid101 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p101, dist101, color101);
		float3 p111 = worldPos + make_float3(P, P, P); float dist111; uchar3 color111; bool valid111 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p111, dist111, color111);

		if (!valid000 || !valid100 || !valid010 || !valid001 || !valid110 || !valid011 || !valid101 || !valid111) return;
		if (dist010 < isolevel) cubeindex += 1;
		if (dist110 < isolevel) cubeindex += 2;
		if (dist100 < isolevel) cubeindex += 4;
		if (dist000 < isolevel) cubeindex += 8;
		if (dist011 < isolevel) cubeindex += 16;
		if (dist111 < isolevel) cubeindex += 32;
		if (dist101 < isolevel) cubeindex += 64;
		if (dist001 < isolevel) cubeindex += 128;

		const float thres = params.m_threshMarchingCubes;
		float distArray[] = { dist000, dist100, dist010, dist001, dist110, dist011, dist101, dist111 };
		for (uint k = 0; k < 8; k++) {
			for (uint l = 0; l < 8; l++) {
				if (distArray[k] * distArray[l] < 0.0f) {
					if (abs(distArray[k]) + abs(distArray[l]) > thres) return;
				}
				else {
					if (abs(distArray[k] - distArray[l]) > thres) return;
				}
			}
		}


		if (abs(dist000) > params.m_threshMarchingCubes2) return;
		if (abs(dist100) > params.m_threshMarchingCubes2) return;
		if (abs(dist010) > params.m_threshMarchingCubes2) return;
		if (abs(dist001) > params.m_threshMarchingCubes2) return;
		if (abs(dist110) > params.m_threshMarchingCubes2) return;
		if (abs(dist011) > params.m_threshMarchingCubes2) return;
		if (abs(dist101) > params.m_threshMarchingCubes2) return;
		if (abs(dist111) > params.m_threshMarchingCubes2) return;

		Voxel v = hashData.getVoxel(worldPos); //worldPos-P
		int texind = v.texind;
		float3 voxel_pos = hashData.worldToVirtualVoxelPosFloat(worldPos);
		if (texind < 0) return;
		char texDir = texPoolData.d_texPatchDir[texind];
		if (edgeTable[cubeindex] & 1)	vertlist[0] = vertexInterp(hashData, texPoolData, P, texind, texDir, voxel_pos, voxelSize, isolevel, p010, p110, dist010, dist110, v.color, v.color);
		if (edgeTable[cubeindex] & 2)	vertlist[1] = vertexInterp(hashData, texPoolData, P, texind, texDir, voxel_pos, voxelSize, isolevel, p110, p100, dist110, dist100, v.color, v.color);
		if (edgeTable[cubeindex] & 4)	vertlist[2] = vertexInterp(hashData, texPoolData, P, texind, texDir, voxel_pos, voxelSize, isolevel, p100, p000, dist100, dist000, v.color, v.color);
		if (edgeTable[cubeindex] & 8)	vertlist[3] = vertexInterp(hashData, texPoolData, P, texind, texDir, voxel_pos, voxelSize, isolevel, p000, p010, dist000, dist010, v.color, v.color);
		if (edgeTable[cubeindex] & 16)	vertlist[4] = vertexInterp(hashData, texPoolData, P, texind, texDir, voxel_pos, voxelSize, isolevel, p011, p111, dist011, dist111, v.color, v.color);
		if (edgeTable[cubeindex] & 32)	vertlist[5] = vertexInterp(hashData, texPoolData, P, texind, texDir, voxel_pos, voxelSize, isolevel, p111, p101, dist111, dist101, v.color, v.color);
		if (edgeTable[cubeindex] & 64)	vertlist[6] = vertexInterp(hashData, texPoolData, P, texind, texDir, voxel_pos, voxelSize, isolevel, p101, p001, dist101, dist001, v.color, v.color);
		if (edgeTable[cubeindex] & 128)	vertlist[7] = vertexInterp(hashData, texPoolData, P, texind, texDir, voxel_pos, voxelSize, isolevel, p001, p011, dist001, dist011, v.color, v.color);
		if (edgeTable[cubeindex] & 256)	vertlist[8] = vertexInterp(hashData, texPoolData, P, texind, texDir, voxel_pos, voxelSize, isolevel, p010, p011, dist010, dist011, v.color, v.color);
		if (edgeTable[cubeindex] & 512)	vertlist[9] = vertexInterp(hashData, texPoolData, P, texind, texDir, voxel_pos, voxelSize, isolevel, p110, p111, dist110, dist111, v.color, v.color);
		if (edgeTable[cubeindex] & 1024) vertlist[10] = vertexInterp(hashData, texPoolData, P, texind, texDir, voxel_pos, voxelSize, isolevel, p100, p101, dist100, dist101, v.color, v.color);
		if (edgeTable[cubeindex] & 2048) vertlist[11] = vertexInterp(hashData, texPoolData, P, texind, texDir, voxel_pos, voxelSize, isolevel, p000, p001, dist000, dist001, v.color, v.color);

		for (int i = 0; triTable[cubeindex][i] != -1; i += 3)
		{
			Triangle t;
			t.v0 = vertlist[triTable[cubeindex][i + 0]];
			t.v1 = vertlist[triTable[cubeindex][i + 1]];
			t.v2 = vertlist[triTable[cubeindex][i + 2]];

			appendTriangle(t);
		}
	}



	__device__
		Vertex vertexInterp(float isolevel, const float3& p1, const float3& p2, float d1, float d2, const uchar3& c1, const uchar3& c2) const
	{
		Vertex r1; r1.p = p1; r1.c = make_float3(c1.x, c1.y, c1.z) / 255.f;
		Vertex r2; r2.p = p2; r2.c = make_float3(c2.x, c2.y, c2.z) / 255.f;

		if (abs(isolevel - d1) < 0.00001f)		return r1;
		if (abs(isolevel - d2) < 0.00001f)		return r2;
		if (abs(d1 - d2) < 0.00001f)			return r1;

		float mu = (isolevel - d1) / (d2 - d1);

		Vertex res;
		res.p.x = p1.x + mu * (p2.x - p1.x); // Positions
		res.p.y = p1.y + mu * (p2.y - p1.y);
		res.p.z = p1.z + mu * (p2.z - p1.z);

		res.c.x = (float)(c1.x + mu * (float)(c2.x - c1.x)) / 255.f; // Color
		res.c.y = (float)(c1.y + mu * (float)(c2.y - c1.y)) / 255.f;
		res.c.z = (float)(c1.z + mu * (float)(c2.z - c1.z)) / 255.f;

		//texture coordinate

		return res;
	}

	__device__
		bool isInBoxAA(const float3& minCorner, const float3& maxCorner, const float3& pos) const
	{
		if (pos.x < minCorner.x || pos.x > maxCorner.x) return false;
		if (pos.y < minCorner.y || pos.y > maxCorner.y) return false;
		if (pos.z < minCorner.z || pos.z > maxCorner.z) return false;

		return true;
	}

	//#ifdef __CUDACC__
	__device__
		uint append() {
		uint addr = atomicAdd(d_numTriangles, 1);
		//TODO check
		return addr;
	}
	//#endif // __CUDACC__

	//#ifdef __CUDACC__
	__device__
		//#else
		//	__host__
		//#endif
		void appendTriangle(const Triangle& t) {
		if (*d_numTriangles >= d_params->m_maxNumTriangles) {
			*d_numTriangles = d_params->m_maxNumTriangles;
			return; // todo
		}

		//#ifdef __CUDACC__
		uint addr = append();
		//#else		
		//		uint addr = *d_numTriangles;
		//		(*d_numTriangles)++;
		//#endif

		if (addr >= d_params->m_maxNumTriangles) {
			printf("marching cubes exceeded max number of triangles (addr, #tri, max#tri): (%d, %d, %d)\n", addr, *d_numTriangles, d_params->m_maxNumTriangles);
			*d_numTriangles = d_params->m_maxNumTriangles;
			return; // todo
		}

		Triangle& triangle = d_triangles[addr];
		triangle.v0 = t.v0;
		triangle.v1 = t.v1;
		triangle.v2 = t.v2;
		return;
	}
#endif // __CUDACC__

	MarchingCubesParams*	d_params;

	uint*			d_numOccupiedBlocks;
	uint*			d_occupiedBlocks;

	uint*			d_numTriangles;
	Triangle*		d_triangles;

	bool			m_bIsOnGPU;				// the class be be used on both cpu and gpu
};

