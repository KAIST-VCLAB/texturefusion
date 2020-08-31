#pragma once

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"
#include "texturePool.h"

#include "CUDATexUpdateParams.h"

#ifndef __CUDACC__
#include "mLib.h"
#endif

extern __constant__ TexUpdateParams c_texUpdateParams;
extern "C" void updateConstantTexUpdateParams(const TexUpdateParams& params);


struct TexUpdateData {

	__device__ __host__
		TexUpdateData() {

		d_flow = NULL;
		d_flow_colorization = NULL;
		d_alpha = NULL;
		d_alphaHelper = NULL;
		d_gray_source = NULL;
		d_gray_target = NULL;
		
		//mask 
		d_depthMask = NULL;
		d_occlusionMask = NULL;
		d_sourceMask = NULL;
		d_targetMask = NULL;
		d_blendingMask = NULL;

		d_wfieldimage = NULL;
		d_wfieldrender = NULL;
	}

#ifndef __CUDACC__

	void allocate(const TexUpdateParams& params) {

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_wfieldimage, sizeof(float2) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_wfieldrender, sizeof(float2) * params.m_width * params.m_height));

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_flow, sizeof(float2) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_flow_colorization, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_alpha, sizeof(float) * params.m_width * params.m_height));

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_alphaHelper, sizeof(float) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_gray_source, sizeof(float) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_gray_target, sizeof(float) * params.m_width * params.m_height));

		//mask
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthMask, sizeof(float) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_occlusionMask, sizeof(float) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_sourceMask, sizeof(float) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_targetMask, sizeof(float) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_blendingMask, sizeof(float) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_warpingMask, sizeof(float) * params.m_width * params.m_height));

	}

	__host__
		void updateParams(const TexUpdateParams& params) {
			updateConstantTexUpdateParams(params);
	}

	__host__
		void free() {

		MLIB_CUDA_SAFE_FREE(d_flow);
		MLIB_CUDA_SAFE_FREE(d_wfieldimage);
		MLIB_CUDA_SAFE_FREE(d_wfieldrender);
		MLIB_CUDA_SAFE_FREE(d_flow_colorization);
		MLIB_CUDA_SAFE_FREE(d_alpha);
		MLIB_CUDA_SAFE_FREE(d_alphaHelper);
		MLIB_CUDA_SAFE_FREE(d_depthMask);
		MLIB_CUDA_SAFE_FREE(d_sourceMask);
		MLIB_CUDA_SAFE_FREE(d_targetMask);
		MLIB_CUDA_SAFE_FREE(d_blendingMask);
		MLIB_CUDA_SAFE_FREE(d_warpingMask);
		MLIB_CUDA_SAFE_FREE(d_occlusionMask); 
	}

#endif

#ifdef __CUDACC__


#endif

	float2 *d_flow;
	float4 *d_flow_colorization;
	float *d_alpha;
	float *d_alphaHelper;
	float2 *d_invmorph;
	float *d_gray_source;
	float *d_gray_target;

	float2 *d_wfieldrender;
	float2 *d_wfieldimage;

	//masks for depth update, flow estimation and blending
	float *d_depthMask;
	float *d_occlusionMask;
	float *d_sourceMask;	// 
	float *d_targetMask;	// 0909 we donot use this map but just in case.
	float *d_blendingMask;
	float *d_warpingMask;


};