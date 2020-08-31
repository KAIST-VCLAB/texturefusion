#pragma once

#include <cutil_inline.h>
#include <cutil_math.h>

#include "cuda_SimpleMatrixUtil.h"
#include "texturePool.h"
#include "RayCastSDFUtil.h"
#include "VoxelUtilHashSDF.h"
#include "DepthCameraUtil.h"
#include "TexUpdateUtil.h"
#include "cudaDebug.h"
#include "CUDAImageUtil.h"
#include "ICPUtil.h"
#include "modeDefine.h"

#define T_PER_BLOCK 16
#define MINF __int_as_float(0xff800000)
#define FLOAT_EPSILON 0.000001f

extern texture<float, cudaTextureType2D, cudaReadModeElementType> depthTextureRef;
extern texture<float4, cudaTextureType2D, cudaReadModeElementType> colorTextureRef;

__global__ void setMask_kernel(float *d_mask, int image_w, int image_h, int value, int offset) {

	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < image_w * image_h) {

		int x, y;

		x = tidx % image_w;
		y = tidx / image_w;

		if (x < offset || y < offset || image_w - x < offset || image_h - y < offset)
			d_mask[tidx] = value;

	}
}

void setMask(float *d_mask, int image_w, int image_h, int value, int offset) {

	int pixel_n = image_w * image_h;
	int block_pixel_n = (pixel_n + T_PER_BLOCK - 1) / T_PER_BLOCK;

	setMask_kernel << <block_pixel_n, T_PER_BLOCK >> > (d_mask, image_w, image_h, value, offset);



}

__global__ void setMask_kernel(bool *d_mask, int image_w, int image_h, int offset) {

	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < image_w * image_h) {

		int x, y;

		x = tidx % image_w;
		y = tidx / image_w;

		if (x < offset || y < offset || image_w - x < offset || image_h - y < offset)
			d_mask[tidx] = 0;

	}
}

void setMask(bool *d_mask, int image_w, int image_h, int offset) {

	int pixel_n = image_w * image_h;
	int block_pixel_n = (pixel_n + T_PER_BLOCK - 1) / T_PER_BLOCK;

	setMask_kernel << <block_pixel_n, T_PER_BLOCK >> > (d_mask, image_w, image_h, offset);



}

__inline__ __device__ float bilinearSampling(float *d_image, int image_w, int image_h, float x, float y) {

	// pixel center: integer
	// 0 < x < image_w - 1.f
	// 0 < y < image_h - 1.f
	if (x <= 0.f) x = 0.f + FLOAT_EPSILON;
	if (y <= 0.f) y = 0.f + FLOAT_EPSILON;
	if (x >= image_w - 1.f) x = image_w - 1.f - FLOAT_EPSILON;
	if (y >= image_h - 1.f) y = image_h - 1.f - FLOAT_EPSILON;

	float xw = x - floor(x);
	float yw = y - floor(y);

	int pixel_idx = (int)(y)* image_w + (int)(x);
	float color, color00, color01, color10, color11;

	color00 = d_image[pixel_idx];
	color10 = d_image[pixel_idx + 1];
	color01 = d_image[pixel_idx + image_w];
	color11 = d_image[pixel_idx + 1 + image_w];

	color = (1.f - xw) * (1.f - yw) * color00
		+ xw * (1.f - yw) * color10
		+ (1.f - xw) * yw * color01
		+ xw * yw * color11;

	return color;
}

__inline__ __device__ float2 bilinearInterpolation(const float2 *d_image, int image_w, float x, float y) {

	float xw = x - floor(x);
	float yw = y - floor(y);

	int pixel_idx = (int)(y)* image_w + (int)(x);
	float2 color, color00, color01, color10, color11;

	color00 = d_image[pixel_idx];
	color10 = d_image[pixel_idx + 1];
	color01 = d_image[pixel_idx + image_w];
	color11 = d_image[pixel_idx + 1 + image_w];

	color = (1 - xw) * (1 - yw) * color00
		+ xw * (1 - yw) * color10
		+ (1 - xw) * yw * color01
		+ xw * yw * color11;

	return color;
}

__inline__ __device__ float bilinearInterpolationFloat(const float &ld, const float &rd, const float &lu, const float &ru, const float2 &weight) {
	
	return ld * (1.f - weight.x) * (1.f - weight.y) +
		rd * weight.x * (1.f - weight.y) +
		lu * (1.f - weight.x) * (1.f - weight.y) +
		ru * (1.f - weight.x) * (1.f - weight.y);


}

__inline__ __device__ void bilinearInterpolationMotion(const float *d_motion, int image_w, int image_h, float x, float y, float3 &rot, float3 &trans) {

	if (x < 0.f)
		x = 0.001f;
	if (x > image_w - 1.f)
		x = image_w - 1.001f;
	if (y < 0.f)
		y = 0.001f;
	if (y > image_h - 1.f)
		y = image_h - 1.001f;

	float2 weight =  make_float2 (x - floor(x), y - floor(y));

	int pixel_idx = (int)(y)* image_w + (int)(x);
	int pixel_tmpidx;
	float weightval;
	float totalval;

	rot = make_float3(0.f);
	trans = make_float3(0.f);
	totalval = 0.f;

	//ld
	weightval = (1.f - weight.x) * (1.f - weight.y);
	pixel_tmpidx = pixel_idx;
	rot += weightval * make_float3 ( d_motion[6 * (pixel_tmpidx)+0], d_motion[6 * (pixel_tmpidx)+1], d_motion[6 * (pixel_tmpidx)+2]);
	trans += weightval * make_float3 (d_motion[6 * (pixel_tmpidx)+3], d_motion[6 * (pixel_tmpidx)+4], d_motion[6 * (pixel_tmpidx)+5]);
	//rd
	weightval = weight.x * (1.f - weight.y);
	pixel_tmpidx = pixel_idx + 1;
	rot += weightval * make_float3(d_motion[6 * (pixel_tmpidx)+0], d_motion[6 * (pixel_tmpidx)+1], d_motion[6 * (pixel_tmpidx)+2]);
	trans += weightval * make_float3(d_motion[6 * (pixel_tmpidx)+3], d_motion[6 * (pixel_tmpidx)+4], d_motion[6 * (pixel_tmpidx)+5]);
	//lu
	weightval = (1.f - weight.x) * weight.y;
	pixel_tmpidx = pixel_idx + image_w;
	rot += make_float3(d_motion[6 * (pixel_tmpidx)+0], d_motion[6 * (pixel_tmpidx)+1], d_motion[6 * (pixel_tmpidx)+2]);
	trans += make_float3(d_motion[6 * (pixel_tmpidx)+3], d_motion[6 * (pixel_tmpidx)+4], d_motion[6 * (pixel_tmpidx)+5]);

	//ru
	weightval = weight.x * weight.y;
	pixel_tmpidx = pixel_idx + image_w + 1;
	rot += make_float3(d_motion[6 * (pixel_tmpidx)+0], d_motion[6 * (pixel_tmpidx)+1], d_motion[6 * (pixel_tmpidx)+2]);
	trans += make_float3(d_motion[6 * (pixel_tmpidx)+3], d_motion[6 * (pixel_tmpidx)+4], d_motion[6 * (pixel_tmpidx)+5]);
}

__global__ void resetTexHeapKernel(TexPoolData texPoolData)
{
	const TexPoolParams& texPoolParams = c_texPoolParams;
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx == 0) {
		texPoolData.d_heapCounter[0] = texPoolParams.m_numTexturePatches - 1;	//points to the last element of the array
	}


	if (idx < texPoolParams.m_numTexturePatches) {

		idx += 1;

		texPoolData.d_heap[idx] = texPoolParams.m_numTexturePatches - idx + 1;
		uint texturePatchSize = texPoolParams.m_texturePatchSize;
		uint base_idx = idx * texturePatchSize;

		texPoolData.d_texPatchDir[idx] = -1;
		for (uint i = 0; i < texturePatchSize; i++) {
			texPoolData.deleteTexel(base_idx + i);
		}
	}
}

extern "C" void resetTexCUDA(TexPoolData& texPoolData, const TexPoolParams& texPoolParams) {

	{

		//For every texel, rset 

		//resetting the heap and SDF blocks
		const dim3 gridSize((texPoolParams.m_numTexturePatches + (T_PER_BLOCK*T_PER_BLOCK) - 1) / (T_PER_BLOCK*T_PER_BLOCK), 1); //
		const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

		resetTexHeapKernel << <gridSize, blockSize >> >(texPoolData);

		//writeArray <uint>(texPoolData.d_heap, "d_texpheap.txt", texPoolParams.m_numTexturePatches);


#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
	}

}

__device__
float frac(float val) {
	return (val - floorf(val));
}

__device__
float3 frac(const float3& val) {
	return make_float3(frac(val.x), frac(val.y), frac(val.z));
}

__device__
bool trilinearInterpolationSimpleFastFast(const HashData& hash, const float3& pos, float& dist) {
	const float oSet = c_hashParams.m_virtualVoxelSize;
	const float3 posDual = pos - make_float3(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f); // relative position of position
	float3 weight = frac(hash.worldToVirtualVoxelPosFloat(pos)); // a - floor (a)

	dist = 0.0f;
	Voxel v = hash.getVoxel(posDual + make_float3(0.0f, 0.0f, 0.0f)); if (v.weight == 0) return false;     dist += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*v.sdf;
	v = hash.getVoxel(posDual + make_float3(oSet, 0.0f, 0.0f)); if (v.weight == 0) return false;		   dist += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*v.sdf;
	v = hash.getVoxel(posDual + make_float3(0.0f, oSet, 0.0f)); if (v.weight == 0) return false;		   dist += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*v.sdf;
	v = hash.getVoxel(posDual + make_float3(0.0f, 0.0f, oSet)); if (v.weight == 0) return false;		   dist += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *v.sdf;
	v = hash.getVoxel(posDual + make_float3(oSet, oSet, 0.0f)); if (v.weight == 0) return false;		   dist += weight.x *	   weight.y *(1.0f - weight.z)*v.sdf;
	v = hash.getVoxel(posDual + make_float3(0.0f, oSet, oSet)); if (v.weight == 0) return false;		   dist += (1.0f - weight.x)*	   weight.y *	   weight.z *v.sdf;
	v = hash.getVoxel(posDual + make_float3(oSet, 0.0f, oSet)); if (v.weight == 0) return false;		   dist += weight.x *(1.0f - weight.y)*	   weight.z *v.sdf;
	v = hash.getVoxel(posDual + make_float3(oSet, oSet, oSet)); if (v.weight == 0) return false;		   dist += weight.x *	   weight.y *	   weight.z *v.sdf;

	return true;
}

__device__
bool getTSDFValues(const HashData& hash, const float3& pos1, const float3& pos2, float& sdf1, float& sdf2) {

	const float oSet = c_hashParams.m_virtualVoxelSize;
	const float3 posDual = pos1 - make_float3(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f); // relative position of position
	float3 virtualPos1 = hash.worldToVirtualVoxelPosFloat(pos1);
	float3 virtualPos2 = hash.worldToVirtualVoxelPosFloat(pos2);
	float3 weight1 = virtualPos1 - floorf(virtualPos1); // a - floor (a)
	float3 weight2 = virtualPos2 - floorf(virtualPos1); // a - floor (a)
	
	sdf1 = 0;
	sdf2 = 0;
	Voxel v = hash.getVoxel(posDual + make_float3(0.0f, 0.0f, 0.0f)); if (v.weight == 0) return false;     sdf1 += (1.0f - weight1.x)*(1.0f - weight1.y)*(1.0f - weight1.z)*v.sdf; sdf2 += (1.0f - weight2.x)*(1.0f - weight2.y)*(1.0f - weight2.z)*v.sdf;
	v = hash.getVoxel(posDual + make_float3(oSet, 0.0f, 0.0f)); if (v.weight == 0) return false;		   sdf1 += weight1.x *(1.0f - weight1.y)*(1.0f - weight1.z)*v.sdf;   sdf2 += weight2.x *(1.0f - weight2.y)*(1.0f - weight2.z)*v.sdf;
	v = hash.getVoxel(posDual + make_float3(0.0f, oSet, 0.0f)); if (v.weight == 0) return false;		   sdf1 += (1.0f - weight1.x)*	   weight1.y *(1.0f - weight1.z)*v.sdf; sdf2 += (1.0f - weight2.x)*	   weight2.y *(1.0f - weight2.z)*v.sdf;
	v = hash.getVoxel(posDual + make_float3(0.0f, 0.0f, oSet)); if (v.weight == 0) return false;		   sdf1 += (1.0f - weight1.x)*(1.0f - weight1.y)*	   weight1.z *v.sdf; sdf2 += (1.0f - weight2.x)*(1.0f - weight2.y)*	   weight2.z *v.sdf;
	v = hash.getVoxel(posDual + make_float3(oSet, oSet, 0.0f)); if (v.weight == 0) return false;		   sdf1 += weight1.x *	   weight1.y *(1.0f - weight1.z)*v.sdf; sdf2 += weight2.x *	   weight2.y *(1.0f - weight2.z)*v.sdf;
	v = hash.getVoxel(posDual + make_float3(0.0f, oSet, oSet)); if (v.weight == 0) return false;		   sdf1 += (1.0f - weight1.x)*	   weight1.y *	   weight1.z *v.sdf; sdf2 += (1.0f - weight2.x)*	   weight2.y *	   weight2.z *v.sdf;
	v = hash.getVoxel(posDual + make_float3(oSet, 0.0f, oSet)); if (v.weight == 0) return false;		   sdf1 += weight1.x *(1.0f - weight1.y)*	   weight1.z *v.sdf; sdf2 += weight2.x *(1.0f - weight2.y)*	   weight2.z *v.sdf;
	v = hash.getVoxel(posDual + make_float3(oSet, oSet, oSet)); if (v.weight == 0) return false;		   sdf1 += weight1.x *	   weight1.y *	   weight1.z *v.sdf; sdf2 += weight2.x *	   weight2.y *	   weight2.z *v.sdf;

	return true;
}

__device__
float3 gradientForPoint(const HashData& hash, const float3& pos)
{
	const float voxelSize = c_hashParams.m_virtualVoxelSize*0.5;
	float3 offset = make_float3(voxelSize, voxelSize, voxelSize);
	bool valid = true;
	float distp00; valid &= trilinearInterpolationSimpleFastFast(hash, pos - make_float3(0.5f*offset.x, 0.0f, 0.0f), distp00);
	float dist0p0; valid &= trilinearInterpolationSimpleFastFast(hash, pos - make_float3(0.0f, 0.5f*offset.y, 0.0f), dist0p0);
	float dist00p; valid &= trilinearInterpolationSimpleFastFast(hash, pos - make_float3(0.0f, 0.0f, 0.5f*offset.z), dist00p);

	float dist100; valid &= trilinearInterpolationSimpleFastFast(hash, pos + make_float3(0.5f*offset.x, 0.0f, 0.0f), dist100);
	float dist010; valid &= trilinearInterpolationSimpleFastFast(hash, pos + make_float3(0.0f, 0.5f*offset.y, 0.0f), dist010);
	float dist001; valid &= trilinearInterpolationSimpleFastFast(hash, pos + make_float3(0.0f, 0.0f, 0.5f*offset.z), dist001);

	float3 grad = make_float3((distp00 - dist100) / offset.x, (dist0p0 - dist010) / offset.y, (dist00p - dist001) / offset.z);

	float l = length(grad);
	if (l == 0.0f || !valid) {
		return make_float3(0.0f, 0.0f, 0.0f);
	}

	// - two times!!!!
	return -grad / l;
}

__device__ bool isZeroCrossingVoxel(const HashData& hash, const float3& pf, int &texind) {

	const float oSet = c_hashParams.m_virtualVoxelSize+ 0.000001;
	Voxel v;

	int voxelstate = 0;
	bool isUpdated = false;

	v = hash.getVoxel(pf + make_float3(0.0f, 0.0f, 0.0f)); if (v.weight == 0) return false;
	if (v.sdf > 0.f) voxelstate |= 1;
	isUpdated = isUpdated | v.isUpdated;
	v = hash.getVoxel(pf + make_float3(oSet, 0.0f, 0.0f)); if (v.weight == 0) return false;
	if (v.sdf > 0.f) voxelstate |= 2;
	isUpdated = isUpdated | v.isUpdated;
	v = hash.getVoxel(pf + make_float3(0.0f, oSet, 0.0f)); if (v.weight == 0) return false;
	if (v.sdf > 0.f) voxelstate |= 4;
	isUpdated = isUpdated | v.isUpdated;
	v = hash.getVoxel(pf + make_float3(0.0f, 0.0f, oSet)); if (v.weight == 0) return false;
	if (v.sdf > 0.f) voxelstate |= 8;
	isUpdated = isUpdated | v.isUpdated;
	v = hash.getVoxel(pf + make_float3(oSet, oSet, 0.0f)); if (v.weight == 0) return false;
	if (v.sdf > 0.f) voxelstate |= 16;
	isUpdated = isUpdated | v.isUpdated;
	v = hash.getVoxel(pf + make_float3(0.0f, oSet, oSet)); if (v.weight == 0) return false;
	if (v.sdf > 0.f) voxelstate |= 32;
	isUpdated = isUpdated | v.isUpdated;
	v = hash.getVoxel(pf + make_float3(oSet, 0.0f, oSet)); if (v.weight == 0) return false;
	if (v.sdf > 0.f) voxelstate |= 64;
	isUpdated = isUpdated | v.isUpdated;
	v = hash.getVoxel(pf + make_float3(oSet, oSet, oSet)); if (v.weight == 0) return false;
	if (v.sdf > 0.f) voxelstate |= 128;
	isUpdated = isUpdated | v.isUpdated;

	//texind = -2;

	if (voxelstate == 255 || voxelstate == 0 || !isUpdated)
		return false;

	return true;
}

__global__ void updateMasksWithCapturedFrameKernel(const DepthCameraData& cameraData, const float *d_depth, const float4 *d_normal, TexUpdateData texUpdateData) {

	const RayCastParams& rayCastParams = c_rayCastParams;
	const TexUpdateParams& textureUpdateParams = c_texUpdateParams;
	const DepthCameraParams& depthCameraParams = c_depthCameraParams;

	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (0 <= x && x < depthCameraParams.m_imageWidth && 0 <= y && y < depthCameraParams.m_imageHeight ) {

		int index = y * depthCameraParams.m_imageWidth + x;

		texUpdateData.d_depthMask[index] = 0.f;

		//mask out pixels observed in a grazing angle.
		if (d_normal[index].x != MINF) {

			// compute a normal and a dot product between camera direction and normal.
			float3 camDir = normalize(cameraData.kinectProjToCamera(x, y, 1.0f));
			float3 normal = make_float3(d_normal[index]);
			float dotValue = fabsf(dot(camDir, normal));

			// if the dot product is below than a threshold, mask it out.
			if (dotValue <= textureUpdateParams.m_angleThreshold_depth)
				texUpdateData.d_depthMask[index] = 1.f;

		}
		
		//mask out invalid depth pixels
		if (d_depth[index] == MINF) {

			texUpdateData.d_depthMask [ index ] = 1;

		}

		// mask out regions near the screen boundary.
		// It makes updated triangles projected on the screen region.
		int screenBoundaryWidth = 20;

		if (x < screenBoundaryWidth || rayCastParams.m_width - x < screenBoundaryWidth)
			texUpdateData.d_depthMask[index] = 1;

		if (y < screenBoundaryWidth || rayCastParams.m_height - y < screenBoundaryWidth)
			texUpdateData.d_depthMask[index] = 1;
	}
}

extern "C" void updateMasksWithCapturedFrameCUDA(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, const float4 *d_normalmap_sensor, TexUpdateData& texUpdateData) {

	//I dont know but, if we pass data class to the kernel, it raise some access error.
	// So I just change them in float

	//What is the mask size?
	const int width = depthCameraParams.m_imageWidth;
	const int height = depthCameraParams.m_imageHeight;

	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	updateMasksWithCapturedFrameKernel << <gridSize, blockSize >> > (depthCameraData, depthCameraData.d_depthData, d_normalmap_sensor, texUpdateData);

	cutilCheckMsg(__FUNCTION__);
}

__global__ void computeOcclusionMaskKernel(const DepthCameraData& cameraData, float *d_depth, float4 *d_normals, float *d_mask, int screenBoundaryWidth) {

	const RayCastParams& rayCastParams = c_rayCastParams;
	const TexUpdateParams& textureUpdateParams = c_texUpdateParams;

	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;


	if (0 <= x && x < rayCastParams.m_width && 0 <= y && y < rayCastParams.m_height) {

		int index = y * rayCastParams.m_width + x;
		d_mask[index] = 0;

		if (d_normals[index].x != MINF) {

			float3 camDir = normalize(cameraData.kinectProjToCamera(x, y, 1.0f));
			float3 normal = make_float3(d_normals[index]);
			float dotValue = fabsf(dot(camDir, normal));

			if (dotValue <= 0.1)
				d_mask[index] = 1.f;

			float depth = tex2D(depthTextureRef, x, y);
			if (d_depth[index] - depth > 0.20)
				d_mask[index] = 1.f;

		}
		else d_mask[index] = 1.f;

		if (x <= screenBoundaryWidth || rayCastParams.m_width - x <= screenBoundaryWidth)
			d_mask[index] = 1.;

		if (y <= screenBoundaryWidth || rayCastParams.m_height - y <= screenBoundaryWidth)
			d_mask[index] = 1.;
	}
}

__global__ void computeSourceMaskKernel(float *d_validMask, float *d_mask) {

	const RayCastParams& rayCastParams = c_rayCastParams;
	const TexUpdateParams& textureUpdateParams = c_texUpdateParams;
	//const DepthCameraParams& depthCameraParams = c_depthCameraParams;

	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (0 <= x && x < rayCastParams.m_width && 0 <= y && y < rayCastParams.m_height) {

		int index = y* rayCastParams.m_width + x;

		d_mask[index] = 0.;

		const float value = d_validMask[index];

		if (value > 0.1f)
			d_mask[index] = 1.;

	}
}

__global__ void maskSlantRegionKernel(const DepthCameraData& cameraData, float4 *d_normals, float *d_mask){

	const RayCastParams& rayCastParams = c_rayCastParams;
	const TexUpdateParams& textureUpdateParams = c_texUpdateParams;
	//const DepthCameraParams& depthCameraParams = c_depthCameraParams;

	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;


	if (0 <= x && x < rayCastParams.m_width && 0 <= y && y < rayCastParams.m_height) {

		int index = y * rayCastParams.m_width + x;

		////avoid grazing observation angle.
		if (d_normals[index].x != MINF) {

			//compute a ray and compute cosine between camera direction and normal
			float3 camDir = normalize(cameraData.kinectProjToCamera(x, y, 1.0f));
			float3 normal = make_float3(d_normals[index]);
			float dotValue = fabsf(dot(camDir, normal));

			//mask threshold
			if (dotValue <= textureUpdateParams.m_angleThreshold_update)
				d_mask[index] = 1.f;

		}
	}
}

__global__ void computeBlendingMaskKernel(const DepthCameraData& cameraData, float4 *d_normals, float *d_mask) {
	
	const RayCastParams& rayCastParams = c_rayCastParams;
	const TexUpdateParams& textureUpdateParams = c_texUpdateParams;
	//const DepthCameraParams& depthCameraParams = c_depthCameraParams;

	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (0 <= x && x < rayCastParams.m_width && 0 <= y && y < rayCastParams.m_height) {

		int index = y * rayCastParams.m_width + x;

		////avoid grazing observation angle.
		if (d_normals[index].x != MINF) {

			//compute a ray and compute cosine between camera direction and normal
			float3 camDir = normalize(cameraData.kinectProjToCamera(x, y, 1.0f));
			float3 normal = make_float3(d_normals[index]);
			float dotValue = fabsf(dot(camDir, normal));

			//	//mask threshold
			if (dotValue <= textureUpdateParams.m_angleThreshold_update)
				d_mask[index] = 1.f;

		}
		else d_mask[index] = 1.f;
	}

}

extern "C" void computeOcclusionMaskCUDA(const RayCastData &rayCastData, const DepthCameraData depthCameraData, const RayCastParams &rayCastParams, float *d_mask, int screenBoundaryWidth) {

	const dim3 gridSize((rayCastParams.m_width + T_PER_BLOCK - 1) / T_PER_BLOCK, (rayCastParams.m_height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);
	
	computeOcclusionMaskKernel << <gridSize, blockSize >> >(depthCameraData, rayCastData.d_depth, rayCastData.d_normals, d_mask, screenBoundaryWidth);

}

extern "C" void maskSlantRegionCUDA(const RayCastData &rayCastData, const DepthCameraData depthCameraData, const RayCastParams &rayCastParams, float *d_mask) {

	const dim3 gridSize((rayCastParams.m_width + T_PER_BLOCK - 1) / T_PER_BLOCK, (rayCastParams.m_height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	maskSlantRegionKernel << <gridSize, blockSize >> >(depthCameraData, rayCastData.d_normals, d_mask);
	
}

extern "C" void computeSourceMaskCUDA(const RayCastData &rayCastData, const DepthCameraData depthCameraData, const RayCastParams &rayCastParams, float *d_mask) {

	const dim3 gridSize((rayCastParams.m_width + T_PER_BLOCK - 1) / T_PER_BLOCK, (rayCastParams.m_height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeSourceMaskKernel <<<gridSize, blockSize>>>(rayCastData.d_validMask, d_mask);


}

extern "C" void computeTargetMaskCUDA(const RayCastData &rayCastData, const DepthCameraData depthCameraData, const RayCastParams &rayCastParams, float *d_mask) {

	const dim3 gridSize((rayCastParams.m_width + T_PER_BLOCK - 1) / T_PER_BLOCK, (rayCastParams.m_height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

}

extern "C" void computeBlendingMaskCUDA(const RayCastData &rayCastData, const DepthCameraData depthCameraData, const RayCastParams &rayCastParams, float *d_mask){

	const dim3 gridSize((rayCastParams.m_width + T_PER_BLOCK - 1) / T_PER_BLOCK, (rayCastParams.m_height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeBlendingMaskKernel << <gridSize, blockSize >> >(depthCameraData, rayCastData.d_normals, d_mask);



}

__global__ void findZeroCrossingVoxelsKernel(HashData hashData, TexPoolData texPoolData, DepthCameraData cameraData, int  *d_tmp) {

	//#define GLOBALMODE

	const HashParams& hashParams = c_hashParams;
	const DepthCameraParams& cameraParams = c_depthCameraParams;

	//TODO check if we should load this in shared memory
	const HashEntry& entry = hashData.d_hashCompactified[blockIdx.x];

	int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);

	uint i = threadIdx.x;	//inside of an SDF block
	int3 pi = pi_base + make_int3(hashData.delinearizeVoxelIndex(i));
	float3 pf = hashData.virtualVoxelPosToWorld(pi);
	float3 pf_dual = pf; //i- 0.5f * hashParams.m_virtualVoxelSize;

	//center of a voxel
	pf += 0.5f * hashParams.m_virtualVoxelSize;

	//find the projection point in the screen space
	pf = hashParams.m_rigidTransformInverse * pf;
	uint2 screenPos = make_uint2(cameraData.cameraToKinectScreenInt(pf));

#ifdef GLOBALMODE
	__shared__ int localCounter;
	if (threadIdx.x == 0) localCounter = 0;
	__syncthreads();
#endif

	uint idx = entry.ptr + i;

	// For a voxel which is in the screen space 
	if (10.f < screenPos.x && screenPos.x < cameraParams.m_imageWidth - 10.f && 10.f < screenPos.y &&screenPos.y < cameraParams.m_imageHeight - 10.f) {

#ifdef GLOBALMODE
		int addrLocal = -1;
#endif

		// if the voxel is a zero-crossing voxel (contains geometry),
		if (isZeroCrossingVoxel(hashData, pf_dual, hashData.d_SDFBlocks[idx].texind)) {

#ifdef GLOBALMODE
			addrLocal = atomicAdd(&localCounter, 1);
#endif

			//select proejction direction of a texture tile
			float3 normal = gradientForPoint(hashData, pf_dual + 0.5 * hashParams.m_virtualVoxelSize);
			float3 absnormal = fabs(normal);

			if (length(absnormal) > 0.001f) {
				
				char dir = -5;
				if (absnormal.x >= absnormal.y && absnormal.x >= absnormal.z) {
					if (normal.x < 0)
						dir = 0;
					else dir = 1;
				}

				else if (absnormal.y >= absnormal.z) {
					if (normal.y < 0)
						dir = 2;
					else dir = 3;
				}

				else {
					if (normal.z < 0)
						dir = 4;
					else dir = 5;
				}

#ifdef GLOBALMODE
				__shared__ int addrGlobal;
				__shared__ int texAddrGlobal;

				//after finish to count all zerocrossing voxel, do process
				__syncthreads();

				//the first thread store the list
				if (threadIdx.x == 0 && localCounter > 0) {

					addrGlobal = atomicAdd(hashData.d_voxelZeroCrossCounter, localCounter);
					texAddrGlobal = texPoolData.consumeHeap(localCounter);

				}

				//sync
				__syncthreads();

				////assign new tex spaces to voxels
				//if (0 <= 60000) {

				//hashData.d_SDFBlocks[idx].texind = texPoolData.d_heap[texAddrGlobal + addrLocal];
#else
				//add the current voxel in the zero-crossing voxel list.
				int addr = atomicAdd(hashData.d_voxelZeroCrossCounter, 1);
				hashData.d_voxelZeroCross[addr] = pi;

				// assign a texture tile
				int texAddr = texPoolData.consumeHeap();
				hashData.d_SDFBlocks[idx].texind = texAddr;
				texPoolData.d_texPatchDir[texAddr] = dir;

			}
#endif
		}
	}
}

//used
extern "C" unsigned int findZeroCrossingVoxelsCUDA(HashData& hashData, HashParams& hashParams, TexPoolData& texPoolData, const DepthCameraData& cameraData)
{
	const unsigned int threadsPerBlock = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
	const dim3 gridSize(hashParams.m_numOccupiedBlocks, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	unsigned int res = 0;
	
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);

	// for debug
	//cutilSafeCall(cudaMemcpy(&res, hashData.d_voxelZeroCrossCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//std::cout << "zeroCrossingVoxelCounter: " << res <<" "<< hashParams.m_hashNumBuckets * 2 << std::endl;
	//cutilSafeCall(cudaMemcpy(&res, texPoolData.d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//std::cout << "texheapCounter: " << res << std::endl;

	cutilSafeCall(cudaMemset(hashData.d_voxelZeroCrossCounter, 0, sizeof(int)));

	int *d_tmp;
	if (hashParams.m_numOccupiedBlocks > 0) {	//this guard is important if there is no depth in the current frame (i.e., no blocks were allocated)

		findZeroCrossingVoxelsKernel << <gridSize, blockSize >> > (hashData, texPoolData, cameraData, d_tmp);

	}

	//cutilSafeCall(cudaMemcpy(&res, texPoolData.d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//std::cout << "texheapCounter: " << res << std::endl;
	cutilSafeCall(cudaMemcpy(&res, hashData.d_voxelZeroCrossCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//std::cout << "zeroCrossingVoxelCounter: " << res << std::endl;
	return res;

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}
__global__ void deletePreviousTexTriangleKernel(HashData hashData, TexPoolData texPoolData, DepthCameraData cameraData, int *d_tmp) {

	const HashParams& hashParams = c_hashParams;
	const DepthCameraParams& cameraParams = c_depthCameraParams;

	//TODO check if we should load this in shared memory
	const HashEntry& entry = hashData.d_hashCompactified[blockIdx.x];

	//if (entry.ptr == FREE_ENTRY) {
	//	printf("invliad integrate");
	//	return; //should never happen since we did the compactification before
	//}

	int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);

	int didx = blockIdx.x;

	uint i = threadIdx.x;	//inside of an SDF block
	int3 pi = pi_base + make_int3(hashData.delinearizeVoxelIndex(i));
	float3 pf = hashData.virtualVoxelPosToWorld(pi);
	float3 pf_dual = pf; // + 0.5f * hashParams.m_virtualVoxelSize;

						 //find the projection point in the screen space
	pf = hashParams.m_rigidTransformInverse * pf;
	uint2 screenPos = make_uint2(cameraData.cameraToKinectScreenInt(pf));

	//int addrLocal = -1;
	//__shared__ int localCounter;
	//if (threadIdx.x == 0) 
	//	localCounter = 0;
	//__syncthreads();
	//if (threadIdx.x == 0)
	//	d_tmp[didx] = -1;


	if (screenPos.x < cameraParams.m_imageWidth && screenPos.y < cameraParams.m_imageHeight) {	//on screen

																								//entry.ptr : address of a voxel
		uint idx = entry.ptr + i;


		//float depth = g_InputDepth[screenPos];
		float depth = tex2D(depthTextureRef, screenPos.x, screenPos.y);

		if (depth != MINF) { // valid depth and color

			if (depth < hashParams.m_maxIntegrationDistance) {

				float sdf = depth - pf.z;
				float truncation = hashData.getTruncation(depth);
				if (sdf > -truncation) // && depthZeroOne >= 0.0f && depthZeroOne <= 1.0f) //check if in truncation range should already be made in depth map computation
				{
					//						if (isZeroCrossingVoxel(hashData, pf)) {

					if (hashData.d_SDFBlocks[idx].prev_texind > 0 && hashData.d_SDFBlocks[idx].prev_texind != hashData.d_SDFBlocks[idx].texind) {

						texPoolData.appendHeap(hashData.d_SDFBlocks[idx].prev_texind);
						hashData.d_SDFBlocks[idx].prev_texind = -1;

					}
				}
			}
			//			}
			//__shared__ int addrGlobal;

			//__syncthreads();

			//if (threadIdx.x == 0)
			//	addrGlobal = texPoolData.appendHeapV(localCounter);

			//__syncthreads();

			//if (addrLocal != -1) {
			//	texPoolData.d_heap[addrGlobal + addrLocal] = hashData.d_SDFBlocks[idx].prev_texind;
			//	hashData.d_SDFBlocks[idx].prev_texind = -1;
			//}
		}
	}
}

extern "C" void deletePreviousTexTriangleCUDA(HashData& hashData, HashParams& hashParams, TexPoolData& texPoolData, const DepthCameraData& cameraData)
{
	const unsigned int threadsPerBlock = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
	const dim3 gridSize(hashParams.m_numOccupiedBlocks, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	unsigned int res = 0;
	int *d_tmp;

	if (hashParams.m_numOccupiedBlocks > 0) {	//this guard is important if there is no depth in the current frame (i.e., no blocks were allocated)

												//cudaMalloc(&d_tmp, sizeof(int) * hashParams.m_numOccupiedBlocks);
		deletePreviousTexTriangleKernel << <gridSize, blockSize >> >(hashData, texPoolData, cameraData, d_tmp);

		cutilSafeCall(cudaMemcpy(&res, texPoolData.d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));

	}

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__device__
float findIntersectionLinear(float tNear, float tFar, float dNear, float dFar)
{
	return tNear + (dNear / (dNear - dFar))*(tFar - tNear);
}

////////////////////////////////////////////////////////////////////////////////////

__inline__ __device__ float computeBlendingWeight(float depth, float3 &normal, float3 &view, const float zmin,const float sigmaAngle, const float sigmaArea) {
	float depthFactor = min(zmin / depth,1.f);
	float cosFactor =  max(0.f, -dot(normal, view));
	float areaFactor = max(0.f, depthFactor * depthFactor * cosFactor);
	
	cosFactor = 1.f - cosFactor;
	areaFactor = 1.f - areaFactor;

	return exp(-(areaFactor * areaFactor) / (sigmaArea * sigmaArea)) * exp(-(cosFactor * cosFactor) / (sigmaAngle * sigmaAngle));
}

__global__ void texUpdateFromImageKernel(HashData hashData, TexUpdateData texUpdateData, TexPoolData texPoolData, RayCastData rayCastData, DepthCameraData cameraData, float *d_mask) {

	const HashParams& hashParams = c_hashParams;
	const TexPoolParams& texPoolParams = c_texPoolParams;
	const DepthCameraParams& cameraParams = c_depthCameraParams;
	const TexUpdateParams& texUpdateParams = c_texUpdateParams;

	//global index of threads
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float weight, weight1, weight2;
	float4 color, color1, color2;
	weight = 0.f;
	weight1 = 0.f;
	weight2 = 0.f;

	if (index < texPoolParams.m_texturePatchSize * hashData.d_voxelZeroCrossCounter[0]) {

		int zerocross_ind = index / texPoolParams.m_texturePatchSize;
		int local_texel_ind = index % texPoolParams.m_texturePatchSize;

		if (zerocross_ind < hashParams.m_numZeroCrossVoxels) {

			// here we can wrap this up as one function.
			int3 voxel_ind = hashData.d_voxelZeroCross[zerocross_ind];
			Voxel v = hashData.getVoxel(voxel_ind);
			int texture_ind = v.texind;

			int texel_addr = (texture_ind * texPoolParams.m_texturePatchSize) + local_texel_ind;

			char texDir = texPoolData.getTexDir(texture_ind);
			float3 texWorldDir = texPoolData.getWorldTexDir(texDir);

			//virtualVoxelPosToWorld
			uint2 uv = texPoolData.delinearizeTexelIndex(local_texel_ind);
			float3 virtual_local_pos = texPoolData.uvToVirtualLocalVoxelPos(uv, texDir);
			float3 world_start_pos = (make_float3(voxel_ind) + virtual_local_pos + 0.0001) * c_hashParams.m_virtualVoxelSize;

			float alpha = texPoolData.d_texPatches[texel_addr].alpha;
			if (alpha > -1.f) {
				float3 world_pos = world_start_pos + alpha * texWorldDir* c_hashParams.m_virtualVoxelSize;

				//Bring color from the current image
				float3 cam_pos = hashParams.m_rigidTransformInverse * world_pos;
				uint2 screenPos = make_uint2(cameraData.cameraToKinectScreenInt(cam_pos));
				float depthRender = cam_pos.z;

				float3 normal = gradientForPoint(hashData, world_pos);

				//Bring color from previous texture
				float3 camDir = normalize(cam_pos); // float3 camDir = normalize(cameraData.kinectProjToCamera(x, y, 1.0f));
				float4 w = hashParams.m_rigidTransform * make_float4(camDir, 0.0f);
				float3 worldDir = normalize(make_float3(w.x, w.y, w.z));

				if (0 <= screenPos.x && screenPos.x < cameraParams.m_imageWidth - 1 && 0 <= screenPos.y && screenPos.y < cameraParams.m_imageHeight - 1) {//on screen

					float maskval = 0.f;

					maskval = bilinearSampling(d_mask, cameraParams.m_imageWidth, cameraParams.m_imageHeight, screenPos.x, screenPos.y);


					//float depthCapture = tex2D(depthTextureRef, screenPos.x, screenPos.y);

					// it has valid depth value and the difference between a depth map and geometry should be under a threshold.
					//if (depthCapture != MINF) {
					//float2 pos = bilinearInterpolation(texUpdateData.d_wfieldimage, cameraParams.m_imageWidth, screenPos.x, screenPos.y);
					float2 pos = make_float2( screenPos.x, screenPos.y);

					float occlusionFactor = maskval;
					float occlusionWeight = 1 - occlusionFactor;

					if (occlusionWeight > 0.0001) {
						float totalWeight = computeBlendingWeight(depthRender, normal, worldDir, cameraParams.m_sensorDepthWorldMin, texUpdateParams.m_sigma_angle, texUpdateParams.m_sigma_area) * texUpdateParams.m_integrationWeightSample;

						//angle mask
						if (cameraData.d_colorData) {

							if (texPoolData.d_texPatches[texel_addr].alpha > -1.f) {
								color1 = make_float4(texPoolData.d_texPatches[texel_addr].color.x, texPoolData.d_texPatches[texel_addr].color.y, texPoolData.d_texPatches[texel_addr].color.z, 255.f);
								weight1 = texPoolData.d_texPatches[texel_addr].weight;
							}


							//texPoolData.d_texPatches[texel_addr].color = make_uchar3(255, 0, 0);


							color2 = tex2D(colorTextureRef, pos.x + 0.5, pos.y + 0.5);

							//texUpdateParams.integrationWeightSample
							weight2 = max(totalWeight, 1.f) *occlusionWeight;
							color2 *= 255;

		

							if (weight1 + weight2 > 0.f) {

#ifdef UPDATE_WEIGHTBLENDING
								color = (weight1 * color1 + weight2 * color2) / (weight1 + weight2);
								//color = make_float4(1, 0, 0, 1);
								weight = fminf(weight1 + weight2, texUpdateParams.m_integrationWeightMax);

								texPoolData.d_texPatches[texel_addr].color = make_uchar3(
									color.x, color.y, color.z);
								texPoolData.d_texPatches[texel_addr].color_dummy = make_uchar3(
									color.x, color.y, color.z);
								//	texPoolData.d_texPatches[texel_addr].color = make_uchar3(255, 0, 0);
								texPoolData.d_texPatches[texel_addr].weight = weight + 0.5;
#endif
#ifdef UPDATE_WINNERTAKESALL
								if (weight2 > weight1) {

									color = color2;
									//color = make_float4(1, 0, 0, 1);
									weight = weight2;

									texPoolData.d_texPatches[texel_addr].color = make_uchar3(
										color.x, color.y, color.z);
									texPoolData.d_texPatches[texel_addr].color_dummy = make_uchar3(
										color.x, color.y, color.z);
									//	texPoolData.d_texPatches[texel_addr].color = make_uchar3(255, 0, 0);
									texPoolData.d_texPatches[texel_addr].weight = weight + 0.5;

								}
#endif
							}
							//	return;
					//	}
						}
					}
				}
			}
		}
	}
}

__global__ void texUpdateFromImageHalfKernel(HashData hashData, TexUpdateData texUpdateData, TexPoolData texPoolData, RayCastData rayCastData, DepthCameraData cameraData, float *d_mask) {

	const HashParams& hashParams = c_hashParams;
	const TexPoolParams& texPoolParams = c_texPoolParams;
	const DepthCameraParams& cameraParams = c_depthCameraParams;
	const TexUpdateParams& texUpdateParams = c_texUpdateParams;

	//global index of threads
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float weight, weight1, weight2;
	float4 color, color1, color2;
	weight = 0.f;
	weight1 = 0.f;
	weight2 = 0.f;

	if (index < texPoolParams.m_texturePatchSize * hashData.d_voxelZeroCrossCounter[0]) {

		int zerocross_ind = index / texPoolParams.m_texturePatchSize;
		int local_texel_ind = index % texPoolParams.m_texturePatchSize;

		if (zerocross_ind < hashParams.m_numZeroCrossVoxels) {

			// here we can wrap this up as one function.
			int3 voxel_ind = hashData.d_voxelZeroCross[zerocross_ind];
			Voxel v = hashData.getVoxel(voxel_ind);
			int texture_ind = v.texind;

			int texel_addr = (texture_ind * texPoolParams.m_texturePatchSize) + local_texel_ind;

			char texDir = texPoolData.getTexDir(texture_ind);
			float3 texWorldDir = texPoolData.getWorldTexDir(texDir);

			//virtualVoxelPosToWorld
			uint2 uv = texPoolData.delinearizeTexelIndex(local_texel_ind);
			float3 virtual_local_pos = texPoolData.uvToVirtualLocalVoxelPos(uv, texDir);
			float3 world_start_pos = (make_float3(voxel_ind) + virtual_local_pos + 0.0001) * c_hashParams.m_virtualVoxelSize;

			float alpha = texPoolData.d_texPatches[texel_addr].alpha;
			if (alpha > -1.f) {
				float3 world_pos = world_start_pos + alpha * texWorldDir* c_hashParams.m_virtualVoxelSize;

				//Bring color from the current image
				float3 cam_pos = hashParams.m_rigidTransformInverse * world_pos;
				uint2 screenPos = make_uint2(cameraData.cameraToKinectScreenInt(cam_pos));
				float depthRender = cam_pos.z;

				float3 normal = gradientForPoint(hashData, world_pos);

				//Bring color from previous texture
				float3 camDir = normalize(cam_pos); // float3 camDir = normalize(cameraData.kinectProjToCamera(x, y, 1.0f));
				float4 w = hashParams.m_rigidTransform * make_float4(camDir, 0.0f);
				float3 worldDir = normalize(make_float3(w.x, w.y, w.z));

				if (0 <= screenPos.x && screenPos.x < cameraParams.m_imageWidth - 1 && 0 <= screenPos.y && screenPos.y < cameraParams.m_imageHeight - 1) {//on screen

					float maskval = 0.f;

					maskval = bilinearSampling(d_mask, cameraParams.m_imageWidth, cameraParams.m_imageHeight, screenPos.x, screenPos.y);

					if (maskval < 0.1) {

						//float depthCapture = tex2D(depthTextureRef, screenPos.x, screenPos.y);

						// it has valid depth value and the difference between a depth map and geometry should be under a threshold.
						//if (depthCapture != MINF) {
						float2 pos = bilinearInterpolation(texUpdateData.d_wfieldimage, cameraParams.m_imageWidth, screenPos.x, screenPos.y);
						
						//Angle threshold
						//cosWeight = max((cosWeight - c_texUpdateParams.m_angleThreshold_update) / (1. - c_texUpdateParams.m_angleThreshold_update), 0.f);
						//cos^4
						//cosWeight *= cosWeight;
						//cosWeight *= cosWeight;

						//angle mask
						if (cameraData.d_colorData) {

							if (texPoolData.d_texPatches[texel_addr].alpha >-1.f) {
								color1 = make_float4(texPoolData.d_texPatches[texel_addr].color.x, texPoolData.d_texPatches[texel_addr].color.y, texPoolData.d_texPatches[texel_addr].color.z, 255.f);
								weight1 = texPoolData.d_texPatches[texel_addr].weight;
							}


							//texPoolData.d_texPatches[texel_addr].color = make_uchar3(255, 0, 0);


							color2 = tex2D(colorTextureRef, pos.x + 0.5, pos.y + 0.5);

							//texUpdateParams.integrationWeightSample
							//weight2 = max(totalWeight, 1.f);
							color2 *= 255;


							if (weight1 + weight2 > 0.f) {

#ifdef UPDATE_WEIGHTBLENDING
								color = (0.5 * color1 + 0.5 * color2) ;
								//color = make_float4(1, 0, 0, 1);
								weight = weight1;

								texPoolData.d_texPatches[texel_addr].color = make_uchar3(
									color.x, color.y, color.z);
								texPoolData.d_texPatches[texel_addr].color_dummy = make_uchar3(
									color.x, color.y, color.z);
								//	texPoolData.d_texPatches[texel_addr].color = make_uchar3(255, 0, 0);
								texPoolData.d_texPatches[texel_addr].weight = weight + 0.5;
#endif
#ifdef UPDATE_WINNERTAKESALL
								if (weight2 > weight1) {

									color = color2;
									//color = make_float4(1, 0, 0, 1);
									weight = weight2;

									texPoolData.d_texPatches[texel_addr].color = make_uchar3(
										color.x, color.y, color.z);
									texPoolData.d_texPatches[texel_addr].color_dummy = make_uchar3(
										color.x, color.y, color.z);
									//	texPoolData.d_texPatches[texel_addr].color = make_uchar3(255, 0, 0);
									texPoolData.d_texPatches[texel_addr].weight = weight + 0.5;

								}
#endif
							}
							//	return;
							//	}
						}
					}
				}
			}
		}
	}
}

__global__ void texUpdateFromImageOnlyKernel(HashData hashData, TexUpdateData texUpdateData, TexPoolData texPoolData, RayCastData rayCastData, DepthCameraData cameraData, float *d_mask) {

	const HashParams& hashParams = c_hashParams;
	const TexPoolParams& texPoolParams = c_texPoolParams;
	const DepthCameraParams& cameraParams = c_depthCameraParams;
	const TexUpdateParams& texUpdateParams = c_texUpdateParams;

	//global index of threads
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float weight, weight1, weight2;
	float4 color, color1, color2;
	weight = 0.f;
	weight1 = 0.f;
	weight2 = 0.f;

	if (index < texPoolParams.m_texturePatchSize * hashData.d_voxelZeroCrossCounter[0]) {

		int zerocross_ind = index / texPoolParams.m_texturePatchSize;
		int local_texel_ind = index % texPoolParams.m_texturePatchSize;

		if (zerocross_ind < hashParams.m_numZeroCrossVoxels) {

			// here we can wrap this up as one function.
			int3 voxel_ind = hashData.d_voxelZeroCross[zerocross_ind];
			Voxel v = hashData.getVoxel(voxel_ind);
			int texture_ind = v.texind;

			int texel_addr = (texture_ind * texPoolParams.m_texturePatchSize) + local_texel_ind;

			char texDir = texPoolData.getTexDir(texture_ind);
			float3 texWorldDir = texPoolData.getWorldTexDir(texDir);

			//virtualVoxelPosToWorld
			uint2 uv = texPoolData.delinearizeTexelIndex(local_texel_ind);
			float3 virtual_local_pos = texPoolData.uvToVirtualLocalVoxelPos(uv, texDir);
			float3 world_start_pos = (make_float3(voxel_ind) + virtual_local_pos + 0.0001) * c_hashParams.m_virtualVoxelSize;

			float alpha = texPoolData.d_texPatches[texel_addr].alpha;
			if (alpha > -1.f) {
				float3 world_pos = world_start_pos + alpha * texWorldDir * c_hashParams.m_virtualVoxelSize;

				//Bring color from the current image
				float3 cam_pos = hashParams.m_rigidTransformInverse * world_pos;
				uint2 screenPos = make_uint2(cameraData.cameraToKinectScreenInt(cam_pos));
				float depthRender = cam_pos.z;

				float3 normal = gradientForPoint(hashData, world_pos);

				//Bring color from previous texture
				float3 camDir = normalize(cam_pos); // float3 camDir = normalize(cameraData.kinectProjToCamera(x, y, 1.0f));
				float4 w = hashParams.m_rigidTransform * make_float4(camDir, 0.0f);
				float3 worldDir = normalize(make_float3(w.x, w.y, w.z));
				uint pixel_idx = screenPos.y * cameraParams.m_imageWidth + screenPos.x;

				if (screenPos.x < cameraParams.m_imageWidth && screenPos.y < cameraParams.m_imageHeight) {//on screen

					float maskval = 0.f;

					maskval = bilinearSampling(d_mask, cameraParams.m_imageWidth, cameraParams.m_imageHeight, screenPos.x, screenPos.y);

					if (maskval < 0.1) {

						float depthCapture = tex2D(depthTextureRef, screenPos.x, screenPos.y);

						float totalWeight = computeBlendingWeight(depthRender, normal, worldDir, cameraParams.m_sensorDepthWorldMin, texUpdateParams.m_sigma_angle, texUpdateParams.m_sigma_area) * texUpdateParams.m_integrationWeightSample;

						//angle mask
						if (cameraData.d_colorData) {
							
							weight1 = 0.f;

							color2 = tex2D(colorTextureRef, screenPos.x, screenPos.y);
							//texUpdateParams.integrationWeightSample
							color2 *= 255;
							weight2 = max(totalWeight, 0.0f);

							if (weight1 + weight2 > 0.f) {
								color = color2;
								weight = weight2;

								texPoolData.d_texPatches[texel_addr].color = make_uchar3(color.x, color.y, color.z);
								//								texPoolData.d_texPatches[texel_addr].color = make_uchar3(255, 0, 0);
								texPoolData.d_texPatches[texel_addr].weight = weight;
							}
							//	return;
						}
						//}
					}
					//}
				}
			}
		}
	}
}

__global__ void texUpdateFromPreviousTextureKernel(HashData hashData, TexPoolData texPoolData, RayCastData rayCastData, DepthCameraData cameraData) {

	const HashParams& hashParams = c_hashParams;
	const TexPoolParams& texPoolParams = c_texPoolParams;
	const DepthCameraParams& cameraParams = c_depthCameraParams;
	const TexUpdateParams& texUpdateParams = c_texUpdateParams;

	//global index of threads
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < texPoolParams.m_texturePatchSize * hashData.d_voxelZeroCrossCounter[0]) {

		int zerocross_ind = index / texPoolParams.m_texturePatchSize;
		int local_texel_ind = index % texPoolParams.m_texturePatchSize;

		if (zerocross_ind < hashParams.m_numZeroCrossVoxels) {

			int3 voxel_ind = hashData.d_voxelZeroCross[zerocross_ind];
			Voxel v = hashData.getVoxel(voxel_ind);
			int texture_ind = v.texind;

			int texel_addr = (texture_ind * texPoolParams.m_texturePatchSize) + local_texel_ind;

			char texDir = texPoolData.getTexDir(texture_ind);
			float3 texWorldDir = texPoolData.getWorldTexDir(texDir);

			//compute a corresponding 3d position to a texel.
			uint2 uv = texPoolData.delinearizeTexelIndex(local_texel_ind);
			float3 virtual_local_pos = texPoolData.uvToVirtualLocalVoxelPos(uv, texDir);
			float3 world_start_pos = (make_float3(voxel_ind) + virtual_local_pos + 0.0001) * c_hashParams.m_virtualVoxelSize;

			float alpha = texPoolData.d_texPatches[texel_addr].alpha;

			if (alpha > -1.f) {
				float3 world_pos = world_start_pos + alpha * texWorldDir * c_hashParams.m_virtualVoxelSize;

#ifdef TEXTRANSFER_NORMAL
				float3 normal = -gradientForPoint(hashData, world_pos);
#endif
				// find the corresponding point in the previous geometry
				float3 cam_pos = hashParams.m_rigidTransformInverse * world_pos;
				uint2 screenPos = make_uint2(cameraData.cameraToKinectScreenInt(cam_pos));

				if (screenPos.x < cameraParams.m_imageWidth && screenPos.y < cameraParams.m_imageHeight) {	//on screen

					float minInterval = -3 * c_hashParams.m_virtualVoxelSize;
					float maxInterval = 3 * c_hashParams.m_virtualVoxelSize;
					
#ifdef TEXTRANSFER_NORMAL
					// find the intersection point along the line in normal direction
					rayCastData.traverseCoarseGridSimpleSamplePrevTexture(hashData, texPoolData, cameraData, world_pos, normal, texel_addr, minInterval, maxInterval);
#endif
#ifdef TEXTRANSFER_ORTHO
					// find the intersection point along the line in texture projection direction
					rayCastData.traverseCoarseGridSimpleSamplePrevTexture(hashData, texPoolData, cameraData, world_pos, texWorldDir, texel_addr, minInterval, maxInterval);
#endif
				}
			}
		}
	}
}

__global__ void computeTexDepth(HashData hashData, TexPoolData texPoolData, RayCastData rayCastData, DepthCameraData cameraData){

	const HashParams& hashParams = c_hashParams;
	const TexPoolParams& texPoolParams = c_texPoolParams;
	const DepthCameraParams& cameraParams = c_depthCameraParams;
	const TexUpdateParams& texUpdateParams = c_texUpdateParams;

	//global index of threads
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < texPoolParams.m_texturePatchSize * hashData.d_voxelZeroCrossCounter[0]) {

		int zerocross_ind = index / texPoolParams.m_texturePatchSize;
		int local_texel_ind = index % texPoolParams.m_texturePatchSize;

		if (zerocross_ind < hashParams.m_numZeroCrossVoxels) {

			int3 voxel_ind = hashData.d_voxelZeroCross[zerocross_ind];
			Voxel v = hashData.getVoxel(voxel_ind);
			int texture_ind = v.texind;
			int texel_addr = (texture_ind * texPoolParams.m_texturePatchSize) + local_texel_ind;
			char texDir = texPoolData.getTexDir(texture_ind);
			float3 texWorldDir = texPoolData.getWorldTexDir(texDir) * c_hashParams.m_virtualVoxelSize;

			//compute texel location
			uint2 uv = texPoolData.delinearizeTexelIndex(local_texel_ind);
			float3 virtual_local_pos = texPoolData.uvToVirtualLocalVoxelPos(uv, texDir);
			float3 world_start_pos = (make_float3(voxel_ind) + virtual_local_pos + 0.0001) * c_hashParams.m_virtualVoxelSize;

			//get a start and a end value
			float sdf_start;
			float sdf_end;

			//initialize texels in a tile
			texPoolData.d_texPatches[texel_addr].color = make_uchar3(0, 0, 0);
			texPoolData.d_texPatches[texel_addr].color_dummy = make_uchar3(0, 0, 0);
			texPoolData.d_texPatches[texel_addr].weight = 0.f;
			texPoolData.d_texPatches[texel_addr].alpha = -10.f;

			//store depth offset for each texel
			if (getTSDFValues(hashData, world_start_pos, world_start_pos + texWorldDir, sdf_start, sdf_end)) {

				//find intersection point
				float c;
				c = findIntersectionLinear(0.f, 1.f, sdf_start, sdf_end);
				float3 world_pos = world_start_pos + c * texWorldDir;
				texPoolData.d_texPatches[texel_addr].alpha = c;

			}
		}
	}
}

extern"C" void computeTexDepthCUDA(HashData& hashData, const HashParams& hashParams, TexPoolData& texPoolData, const TexPoolParams& texPoolParams, RayCastData& rayCastData, const DepthCameraData& cameraData) {
		
	const uint numTexels = hashParams.m_numZeroCrossVoxels * texPoolParams.m_texturePatchSize;

	const dim3 gridSize((numTexels + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK);

	computeTexDepth << <gridSize, blockSize >> >(
		hashData, texPoolData, rayCastData, cameraData
		);

}

extern"C" void texUpdateFromPreviousTextureCUDA(HashData& hashData, const HashParams& hashParams, TexPoolData& texPoolData, const TexPoolParams& texPoolParams, RayCastData& rayCastData, const DepthCameraData& cameraData) {

	const uint numTexels = hashParams.m_numZeroCrossVoxels * texPoolParams.m_texturePatchSize;
	const dim3 gridSize((numTexels + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK);

	texUpdateFromPreviousTextureKernel << <gridSize, blockSize >> > (
		hashData, texPoolData, rayCastData, cameraData
		);

}

extern"C" void texUpdateFromImageCUDA(HashData& hashData, const HashParams& hashParams, TexUpdateData& texUpdateData, TexPoolData& texPoolData, const TexPoolParams& texPoolParams, RayCastData& rayCastData, const DepthCameraData& cameraData, float *d_blending_mask) {

	//const uint numTexels = hashParams.m_numZeroCrossVoxels * texPoolParams.m_texturePatchSize;
	//const dim3 gridSize;
	//const dim3 blockSize;

	//const dim3 gridSize ( hashParams.m_numZeroCrossVoxels );
	//const dim3 blockSize ( texPoolParams.m_texturePatchSize );

	const uint numTexels = hashParams.m_numZeroCrossVoxels * texPoolParams.m_texturePatchSize;

	const dim3 gridSize((numTexels + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK);

	texUpdateFromImageKernel << <gridSize, blockSize >> > (
		hashData, texUpdateData, texPoolData, rayCastData, cameraData, d_blending_mask
		);
}

extern"C" void texUpdateFromImageHalfCUDA(HashData& hashData, const HashParams& hashParams, TexUpdateData& texUpdateData, TexPoolData& texPoolData, const TexPoolParams& texPoolParams, RayCastData& rayCastData, const DepthCameraData& cameraData, float *d_blending_mask) {

	//const uint numTexels = hashParams.m_numZeroCrossVoxels * texPoolParams.m_texturePatchSize;
	//const dim3 gridSize;
	//const dim3 blockSize;

	//const dim3 gridSize ( hashParams.m_numZeroCrossVoxels );
	//const dim3 blockSize ( texPoolParams.m_texturePatchSize );

	const uint numTexels = hashParams.m_numZeroCrossVoxels * texPoolParams.m_texturePatchSize;

	const dim3 gridSize((numTexels + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK);

	texUpdateFromImageHalfKernel << <gridSize, blockSize >> > (
		hashData, texUpdateData, texPoolData, rayCastData, cameraData, d_blending_mask
		);
}

__global__ void texUpdateFromImageWithCameraMotionMapKernel(HashData hashData, TexUpdateData texUpdateData, TexPoolData texPoolData, RayCastData rayCastData, DepthCameraData cameraData, float *d_x_map, float *d_occlusionweight) {
	//		hashData, texUpdateData, texPoolData, rayCastData, cameraData, d_motion_map

	const HashParams& hashParams = c_hashParams;
	const TexPoolParams& texPoolParams = c_texPoolParams;
	const DepthCameraParams& cameraParams = c_depthCameraParams;
	const TexUpdateParams& texUpdateParams = c_texUpdateParams;

	//global index of threads
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float weight, weight1, weight2;
	float4 color, color1, color2;
	weight = 0.f;
	weight1 = 0.f;
	weight2 = 0.f;

	if (index < texPoolParams.m_texturePatchSize * hashData.d_voxelZeroCrossCounter[0]) {

		int zerocross_ind = index / texPoolParams.m_texturePatchSize;
		int local_texel_ind = index % texPoolParams.m_texturePatchSize;

		if (zerocross_ind < hashParams.m_numZeroCrossVoxels) {

			// here we can wrap this up as one function.
			int3 voxel_ind = hashData.d_voxelZeroCross[zerocross_ind];
			Voxel v = hashData.getVoxel(voxel_ind);
			int texture_ind = v.texind;

			if (texture_ind > 0) {

				int texel_addr = (texture_ind * texPoolParams.m_texturePatchSize) + local_texel_ind;

				char texDir = texPoolData.getTexDir(texture_ind);
				float3 texWorldDir = texPoolData.getWorldTexDir(texDir);

				//virtualVoxelPosToWorld
				uint2 uv = texPoolData.delinearizeTexelIndex(local_texel_ind);
				float3 virtual_local_pos = texPoolData.uvToVirtualLocalVoxelPos(uv, texDir);
				float3 world_start_pos = (make_float3(voxel_ind) + virtual_local_pos + 0.0001) * c_hashParams.m_virtualVoxelSize;

				float alpha = texPoolData.d_texPatches[texel_addr].alpha;
				if (alpha > -1.f) {
					float3 world_pos = world_start_pos + alpha * texWorldDir* c_hashParams.m_virtualVoxelSize;

					//Bring color from the current image
					// here we have to find projection compute 

					float3 cam_pos = hashParams.m_rigidTransformInverse * world_pos;
					uint2 screenPos = make_uint2(cameraData.cameraToKinectScreenInt(cam_pos));
					float depthRender = cam_pos.z;

					float3 normal = gradientForPoint(hashData, world_pos);

					//Bring color from previous texture
					float3 camDir = normalize(cam_pos); // float3 camDir = normalize(cameraData.kinectProjToCamera(x, y, 1.0f));
					float4 w = hashParams.m_rigidTransform * make_float4(camDir, 0.0f);
					float3 worldDir = normalize(make_float3(w.x, w.y, w.z));

					if (0 <= screenPos.x && screenPos.x < cameraParams.m_imageWidth - 1 && 0 <= screenPos.y && screenPos.y < cameraParams.m_imageHeight - 1) {//on screen

						float3 xRot, xTrans;
						//					bilinearInterpolationMotion(d_x_map, cameraParams.m_imageWidth, cameraParams.m_imageHeight, screenPos.x, screenPos.y, xRot, xTrans);

											//here we can compute the occlusion edge weight

											//d_occlusionweight 
						int index = (int)screenPos.x + (int)screenPos.y *  cameraParams.m_imageWidth;

						//float depthZeroOneValue = cameraData.cameraToKinectProjZ(depthRender);
						//float cosValue = max(0.f, -dot(normal, worldDir));

						//float sigma_depth = texUpdateParams.m_sigma_depth, sigma_angle = texUpdateParams.m_sigma_angle;

						//// inverse value for Gaussian weight (weight at 0 will be 1)
						//cosValue = 1.f - cosValue;

						//float depthWeight = exp(-(depthZeroOneValue * depthZeroOneValue) / (sigma_depth * sigma_depth));
						//float cosWeight = exp(-(cosValue * cosValue) / (sigma_angle * sigma_angle));
						//float occlusionWeight = 1. - occlusionValue;
						//float totalWeight = depthWeight * cosWeight  * texUpdateParams.m_integrationWeightSample;

						//
						//						float occlusionValue = d_occlusionweight[index];
						//float depthZeroOneValue = depthRender / cameraData.cameraToKinectProjZ(depthRender);

						float occlusionFactor = d_occlusionweight[index];
						float occlusionWeight = min(max(1.f - occlusionFactor, 0.f), 1.f);

						if(occlusionWeight > 0.0001f){

						float totalWeight = computeBlendingWeight(depthRender, normal, worldDir, cameraParams.m_sensorDepthWorldMin, texUpdateParams.m_sigma_angle, texUpdateParams.m_sigma_area) * texUpdateParams.m_integrationWeightSample;


						xRot.x = d_x_map[6 * ((int)screenPos.x + (int)screenPos.y *  cameraParams.m_imageWidth) + 0];
						xRot.y = d_x_map[6 * ((int)screenPos.x + (int)screenPos.y *  cameraParams.m_imageWidth) + 1];
						xRot.z = d_x_map[6 * ((int)screenPos.x + (int)screenPos.y *  cameraParams.m_imageWidth) + 2];
						xTrans.x = d_x_map[6 * ((int)screenPos.x + (int)screenPos.y *  cameraParams.m_imageWidth) + 3];
						xTrans.y = d_x_map[6 * ((int)screenPos.x + (int)screenPos.y *  cameraParams.m_imageWidth) + 4];
						xTrans.z = d_x_map[6 * ((int)screenPos.x + (int)screenPos.y *  cameraParams.m_imageWidth) + 5];
						float3x3 rot = evalRMat(xRot);
						float4x4 transform;
						transform = float4x4(rot);
						transform(0, 3) = xTrans.x;
						transform(1, 3) = xTrans.y;
						transform(2, 3) = xTrans.z;
						transform(3, 3) = 1;

						float3 newcam_pos = transform * cam_pos;
						uint2 screenNewPos = make_uint2(cameraData.cameraToKinectScreenInt(newcam_pos));

						if (texPoolData.d_texPatches[texel_addr].alpha >-1.f) {

							color1 = make_float4(texPoolData.d_texPatches[texel_addr].color.x, texPoolData.d_texPatches[texel_addr].color.y, texPoolData.d_texPatches[texel_addr].color.z, 255.f);
							weight1 = texPoolData.d_texPatches[texel_addr].weight;

						}

						if (0 <= screenNewPos.x && screenNewPos.x < cameraParams.m_imageWidth - 1 && 0 <= screenNewPos.y && screenNewPos.y < cameraParams.m_imageHeight - 1) {//on screen

							float depth = tex2D(depthTextureRef, screenPos.x, screenPos.y);
							float truncation = hashData.getTruncation(depth);

							color2 = tex2D(colorTextureRef, screenNewPos.x + 0.5, screenNewPos.y + 0.5);
							if (depth != MINF && abs(depth - depthRender) < truncation) {
								weight2 = max(totalWeight, 1.f) * occlusionWeight;
								color2 *= 255;

								if (weight1 + weight2 > 0.f) {

#ifdef UPDATE_WEIGHTBLENDING
									color = (weight1 * color1 + weight2 * color2) / (weight1 + weight2);
									//color = make_float4(1, 0, 0, 1);
									weight = fminf(weight1 + weight2, texUpdateParams.m_integrationWeightMax);

									texPoolData.d_texPatches[texel_addr].color = make_uchar3(
										color.x, color.y, color.z);
									texPoolData.d_texPatches[texel_addr].color_dummy = make_uchar3(
										color.x, color.y, color.z);
									//	texPoolData.d_texPatches[texel_addr].color = make_uchar3(255, 0, 0);
									texPoolData.d_texPatches[texel_addr].weight = weight + 0.5;
#endif

#ifdef UPDATE_WINNERTAKESALL
									if (weight2 > weight1) {

										color = color2;
										//color = make_float4(1, 0, 0, 1);
										weight = weight2;

										texPoolData.d_texPatches[texel_addr].color = make_uchar3(
											color.x, color.y, color.z);
										texPoolData.d_texPatches[texel_addr].color_dummy = make_uchar3(
											color.x, color.y, color.z);
										//	texPoolData.d_texPatches[texel_addr].color = make_uchar3(255, 0, 0);
										texPoolData.d_texPatches[texel_addr].weight = weight + 0.5;

									}
#endif
								}
							}
						}
					}
				}
			}
		}
	}
}
}

__global__ void texUpdateFromImageWithCameraMotionMapHalfKernel(HashData hashData, TexUpdateData texUpdateData, TexPoolData texPoolData, RayCastData rayCastData, DepthCameraData cameraData, float *d_x_map, float *d_occlusionweight) {
	//		hashData, texUpdateData, texPoolData, rayCastData, cameraData, d_motion_map

	const HashParams& hashParams = c_hashParams;
	const TexPoolParams& texPoolParams = c_texPoolParams;
	const DepthCameraParams& cameraParams = c_depthCameraParams;
	const TexUpdateParams& texUpdateParams = c_texUpdateParams;

	//global index of threads
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	float weight, weight1, weight2;
	float4 color, color1, color2;
	weight = 0.f;
	weight1 = 0.f;
	weight2 = 0.f;

	if (index < texPoolParams.m_texturePatchSize * hashData.d_voxelZeroCrossCounter[0]) {

		int zerocross_ind = index / texPoolParams.m_texturePatchSize;
		int local_texel_ind = index % texPoolParams.m_texturePatchSize;

		if (zerocross_ind < hashParams.m_numZeroCrossVoxels) {

			// here we can wrap this up as one function.
			int3 voxel_ind = hashData.d_voxelZeroCross[zerocross_ind];
			Voxel v = hashData.getVoxel(voxel_ind);
			int texture_ind = v.texind;

			if (texture_ind > 0) {
				

				int texel_addr = (texture_ind * texPoolParams.m_texturePatchSize) + local_texel_ind;

				char texDir = texPoolData.getTexDir(texture_ind);
				float3 texWorldDir = texPoolData.getWorldTexDir(texDir);

				//virtualVoxelPosToWorld
				uint2 uv = texPoolData.delinearizeTexelIndex(local_texel_ind);
				float3 virtual_local_pos = texPoolData.uvToVirtualLocalVoxelPos(uv, texDir);
				float3 world_start_pos = (make_float3(voxel_ind) + virtual_local_pos + 0.0001) * c_hashParams.m_virtualVoxelSize;

				float alpha = texPoolData.d_texPatches[texel_addr].alpha;
				if (alpha > -1.f) {
					float3 world_pos = world_start_pos + alpha * texWorldDir* c_hashParams.m_virtualVoxelSize;

					//Bring color from the current image
					// here we have to find projection compute 

					float3 cam_pos = hashParams.m_rigidTransformInverse * world_pos;
					uint2 screenPos = make_uint2(cameraData.cameraToKinectScreenInt(cam_pos));
					float depthRender = cam_pos.z;

					float3 normal = gradientForPoint(hashData, world_pos);

					//Bring color from previous texture
					float3 camDir = normalize(cam_pos); // float3 camDir = normalize(cameraData.kinectProjToCamera(x, y, 1.0f));
					float4 w = hashParams.m_rigidTransform * make_float4(camDir, 0.0f);
					float3 worldDir = normalize(make_float3(w.x, w.y, w.z));

					if (0 <= screenPos.x && screenPos.x < cameraParams.m_imageWidth - 1 && 0 <= screenPos.y && screenPos.y < cameraParams.m_imageHeight - 1) {//on screen

						float3 xRot, xTrans;
						//					bilinearInterpolationMotion(d_x_map, cameraParams.m_imageWidth, cameraParams.m_imageHeight, screenPos.x, screenPos.y, xRot, xTrans);

						//here we can compute the occlusion edge weight

						//d_occlusionweight 
						int index = (int)screenPos.x + (int)screenPos.y *  cameraParams.m_imageWidth;

						xRot.x = d_x_map[6 * ((int)screenPos.x + (int)screenPos.y *  cameraParams.m_imageWidth) + 0];
						xRot.y = d_x_map[6 * ((int)screenPos.x + (int)screenPos.y *  cameraParams.m_imageWidth) + 1];
						xRot.z = d_x_map[6 * ((int)screenPos.x + (int)screenPos.y *  cameraParams.m_imageWidth) + 2];
						xTrans.x = d_x_map[6 * ((int)screenPos.x + (int)screenPos.y *  cameraParams.m_imageWidth) + 3];
						xTrans.y = d_x_map[6 * ((int)screenPos.x + (int)screenPos.y *  cameraParams.m_imageWidth) + 4];
						xTrans.z = d_x_map[6 * ((int)screenPos.x + (int)screenPos.y *  cameraParams.m_imageWidth) + 5];
						float3x3 rot = evalRMat(xRot);
						float4x4 transform;
						transform = float4x4(rot);
						transform(0, 3) = xTrans.x;
						transform(1, 3) = xTrans.y;
						transform(2, 3) = xTrans.z;
						transform(3, 3) = 1;
						float3 newcam_pos = transform * cam_pos;
						uint2 screenNewPos = make_uint2(cameraData.cameraToKinectScreenInt(newcam_pos));

						if (texPoolData.d_texPatches[texel_addr].alpha >-1.f) {

							color1 = make_float4(texPoolData.d_texPatches[texel_addr].color.x, texPoolData.d_texPatches[texel_addr].color.y, texPoolData.d_texPatches[texel_addr].color.z, 255.f);
							weight1 = texPoolData.d_texPatches[texel_addr].weight;

						}

						if (0 <= screenNewPos.x && screenNewPos.x < cameraParams.m_imageWidth - 1 && 0 <= screenNewPos.y && screenNewPos.y < cameraParams.m_imageHeight - 1) {//on screen

							float depth = tex2D(depthTextureRef, screenNewPos.x, screenNewPos.y);
							float truncation = hashData.getTruncation(depth);

							color2 = tex2D(colorTextureRef, screenNewPos.x + 0.5, screenNewPos.y + 0.5);
							if (depth != MINF && abs(depth - depthRender) < truncation) {
								color2 *= 255;

								if (weight1 + weight2 > 0.f) {

#ifdef UPDATE_WEIGHTBLENDING
									color = (0.5 * color1 + 0.5 * color2);
									//color = make_float4(1, 0, 0, 1);
									weight = weight1;

									texPoolData.d_texPatches[texel_addr].color = make_uchar3(
										color.x, color.y, color.z);
									texPoolData.d_texPatches[texel_addr].color_dummy = make_uchar3(
										color.x, color.y, color.z);
									//	texPoolData.d_texPatches[texel_addr].color = make_uchar3(255, 0, 0);
									texPoolData.d_texPatches[texel_addr].weight = weight + 0.5;
#endif
#ifdef UPDATE_WINNERTAKESALL
									if (weight2 > weight1) {

										color = color2;
										//color = make_float4(1, 0, 0, 1);
										weight = weight2;

										texPoolData.d_texPatches[texel_addr].color = make_uchar3(
											color.x, color.y, color.z);
										texPoolData.d_texPatches[texel_addr].color_dummy = make_uchar3(
											color.x, color.y, color.z);
										//	texPoolData.d_texPatches[texel_addr].color = make_uchar3(255, 0, 0);
										texPoolData.d_texPatches[texel_addr].weight = weight + 0.5;

									}
#endif
								}
							}
						}
					}
				}
			}
		}
	}
}

extern"C" void texUpdateFromImageWithCameraMotionMapCUDA(HashData& hashData, const HashParams& hashParams, TexUpdateData &texUpdateData, TexPoolData& texPoolData, const TexPoolParams& texPoolParams, RayCastData& rayCastData, const DepthCameraData& cameraData, float *d_motion_map, float *d_occlusionweight) {

	const uint numTexels = hashParams.m_numZeroCrossVoxels * texPoolParams.m_texturePatchSize;

	const dim3 gridSize((numTexels + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK);

	texUpdateFromImageWithCameraMotionMapKernel << <gridSize, blockSize >> > (
		hashData, texUpdateData, texPoolData, rayCastData, cameraData, d_motion_map, d_occlusionweight);

}

extern"C" void texUpdateFromImageWithCameraMotionMapHalfCUDA(HashData& hashData, const HashParams& hashParams, TexUpdateData &texUpdateData, TexPoolData& texPoolData, const TexPoolParams& texPoolParams, RayCastData& rayCastData, const DepthCameraData& cameraData, float *d_motion_map, float *d_occlusionweight) {

	const uint numTexels = hashParams.m_numZeroCrossVoxels * texPoolParams.m_texturePatchSize;

	const dim3 gridSize((numTexels + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK);

	texUpdateFromImageWithCameraMotionMapHalfKernel << <gridSize, blockSize >> > (
		hashData, texUpdateData, texPoolData, rayCastData, cameraData, d_motion_map, d_occlusionweight);

}

extern"C" void texUpdateFromImageOnlyCUDA(HashData& hashData, const HashParams& hashParams, TexUpdateData& texUpdateData, TexPoolData& texPoolData, const TexPoolParams& texPoolParams, RayCastData& rayCastData, const DepthCameraData& cameraData,float *d_mask) {

	//const uint numTexels = hashParams.m_numZeroCrossVoxels * texPoolParams.m_texturePatchSize;
	//const dim3 gridSize;
	//const dim3 blockSize;

	//const dim3 gridSize ( hashParams.m_numZeroCrossVoxels );
	//const dim3 blockSize ( texPoolParams.m_texturePatchSize );

	const uint numTexels = hashParams.m_numZeroCrossVoxels * texPoolParams.m_texturePatchSize;

	const dim3 gridSize((numTexels + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK);

	//TexUpdateData a;
	//a.d_colorMask

	texUpdateFromImageOnlyKernel << <gridSize, blockSize >> > (
		hashData, texUpdateData, texPoolData, rayCastData, cameraData, d_mask
		);
}

__global__ void resetAllColorKernel(HashData hashData, TexPoolData texPoolData, TexPoolParams texPoolParams) {


	const HashParams& hashParams = c_hashParams;
	const DepthCameraParams& cameraParams = c_depthCameraParams;

	const HashEntry& entry = hashData.d_hashCompactified[blockIdx.x];

	uint idx = entry.ptr + threadIdx.x;

	int texture_ind = hashData.d_SDFBlocks[idx].texind;

	if (texture_ind >= 0) {

		int texel_addr = (texture_ind * texPoolParams.m_texturePatchSize);

		for (int texel_i = 0; texel_i < texPoolParams.m_texturePatchSize; texel_i++) {
//			texPoolData.d_texPatches[texel_addr + texel_i].color = make_uchar3(255, 255, 255);
			texPoolData.d_texPatches[texel_addr + texel_i].color = make_uchar3(0, 0, 0);
			texPoolData.d_texPatches[texel_addr + texel_i].weight = 0;
		}
	
	}
}

extern"C" void resetColorCUDA(HashData& hashData, const HashParams& hashParams, const TexPoolData& texPoolData, const TexPoolParams& texPoolParams){

	const unsigned int threadsPerBlock = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
	const dim3 gridSize(hashParams.m_numOccupiedBlocks, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	if (hashParams.m_numOccupiedBlocks > 0) {	//this guard is important if there is no depth in the current frame (i.e., no blocks were allocated)
		resetAllColorKernel << <gridSize, blockSize >> >(hashData, texPoolData, texPoolParams);
	}

	//resetAllColorCUDA(hashData, hashParams, m_texPoolData, m_texPoolParams);

}

__global__ void erodeHoleKernel(float* d_output, float* d_input, int radius, unsigned int width, unsigned int height) {

	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	d_output[y*width + x] = 0;

	float sum = 0.f;

	for (int m = x - radius; m <= x + radius; m++)
	{
		for (int n = y - radius; n <= y + radius; n++)
		{
			if (m >= 0 && n >= 0 && m < width && n < height)
			{
				const float currentValue = d_input[n*width + m];
				sum += currentValue;
			}
		}
	}

	if (sum >= 0.95f)
		d_output[y*width + x] = 1.f;

}

extern "C" void erodeHole(float* d_output, float* d_input, int radius, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	erodeHoleKernel << <gridSize, blockSize >> >(d_output, d_input, radius, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

}

__global__ void erodeDistKernel(float* d_output, float* d_input, int radius, float degradeFactor, unsigned int width, unsigned int height) {

	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	const int index = y*width + x;

	d_output[y*width + x] = 0;

	float sum = 0.f;

	for (int m = x - radius; m <= x + radius; m++)
	{
		for (int n = y - radius; n <= y + radius; n++)
		{
			if (m >= 0 && n >= 0 && m < width && n < height)
			{
				const float currentValue = d_input[n*width + m];
//				const float dist = sqrt((float)( (m - x)*(m - x) + (n - y)*(n - y) ));
				const float dist = abs((float)(m - x))+ abs((float)(n - y));
				d_output[ index] = max(d_output[index],currentValue - dist * degradeFactor);
			}
		}
	}	
}

extern "C" void erodeDistCUDA(float* d_output, float* d_input, int radius, float degradeFactor, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	erodeDistKernel << <gridSize, blockSize >> >(d_output, d_input, radius, degradeFactor, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

}

__global__ void 
setAlphaDevice(float* d_output, float* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	d_output[y*width + x] = 0;
	const float valueCenter = d_input[y*width + x];

	if (valueCenter == MINF)
		d_output[y*width + x] = 1.0f;

}

extern "C" void setAlpha(float* d_output, float* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	setAlphaDevice << <gridSize, blockSize >> >(d_output, d_input, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

}

__global__ void
exportTextureImageDevice(uchar *d_output, uint *d_heap, Texel *d_input, unsigned int tex_width, unsigned int tex_height, unsigned int patch_width, unsigned int patch_size, unsigned int num_patches, unsigned int max_num_patches)
{
	const int patch_x = blockIdx.x;
	const int patch_y = blockIdx.y;
	const int patch_idx = patch_y * tex_width + patch_x;
	
	if (patch_idx >= num_patches) return;

	const int texel_x = threadIdx.x;
	const int texel_y = threadIdx.y;

	const int pixel_x = patch_x * patch_width + texel_x;
	const int pixel_y = patch_y * patch_width + texel_y;

	const int pixel_width = tex_width * patch_width;

	const int pixel_idx = pixel_y * pixel_width + pixel_x;
	const int texel_index = d_heap[max_num_patches - patch_idx - 1] * patch_size + texel_y * patch_width + texel_x;
	
	d_output[3 * pixel_idx] = d_input[texel_index].color.x;
	d_output[3 * pixel_idx + 1] = d_input[texel_index].color.y;
	d_output[3 * pixel_idx + 2] = d_input[texel_index].color.z;
}

extern "C" void exportTextureImage(uchar *d_output, uint *d_heap, Texel *d_input, unsigned int tex_width, unsigned int tex_height, unsigned int patch_width, unsigned int patch_size, unsigned int num_patches, unsigned int max_num_patches)
{
	const dim3 gridSize(tex_width, tex_width);
	const dim3 blockSize(patch_width, patch_width);

	exportTextureImageDevice << <gridSize, blockSize >> > (d_output, d_heap, d_input, tex_width, tex_height, patch_width, patch_size, num_patches, max_num_patches);
}
