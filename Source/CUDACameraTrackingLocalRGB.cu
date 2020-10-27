#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"
#include "ICPUtil.h"
#include "CameraTrackingInput.h"

#define THREAD_PER_BLOCK 32

__global__ void renderCorrespondenceLocal_kernel(unsigned int imageWidth, unsigned int imageHeight, float *output, CameraTrackingLocalInput cameraTrackingInput, float3x3 intrinsics, CameraTrackingLocalParameters cameraTrackingParameters, float *d_transforms) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	const int index1D = x;
	const uint2 index = make_uint2(x % imageWidth, x / imageWidth);
	const int nodeW = cameraTrackingParameters.nodeWidth;
	const int nodeH = cameraTrackingParameters.nodeHeight;
	float weight = 0;
	cameraTrackingParameters.offset;

	if (index.x < imageWidth && index.y < imageHeight) {

		float3 pInput = make_float3(cameraTrackingInput.d_inputPos[index1D]);
		float3 nInput = make_float3(cameraTrackingInput.d_inputNormal[index1D]);
		float iInput = cameraTrackingInput.d_inputIntensity[index1D];

		float2 posLocal = (make_float2(index) - cameraTrackingParameters.offset) / cameraTrackingParameters.cellWH;

		if (posLocal.x > nodeW - 1) posLocal.x = nodeW - 1.000001;
		if (posLocal.x < 0) posLocal.x = 0.000001;
		if (posLocal.y > nodeH - 1) posLocal.y = nodeH - 1.000001;
		if (posLocal.y < 0) posLocal.y = 0.000001;

		int2 posi = make_int2(posLocal);
		int nodeind = posi.y * nodeW + posi.x;

		float2 weight = make_float2(posLocal.x - (int)posLocal.x, posLocal.y - (int)posLocal.y);

		float3 ldea, rdea, luea, ruea;
		float3 ldt, rdt, lut, rut;

		ldea = make_float3(d_transforms[6 * nodeind + 0], d_transforms[6 * nodeind + 1], d_transforms[6 * nodeind + 2]);
		ldt = make_float3(d_transforms[6 * nodeind + 3], d_transforms[6 * nodeind + 4], d_transforms[6 * nodeind + 5]);

		rdea = make_float3(0.f);
		luea = make_float3(0.f);
		ruea = make_float3(0.f);
		rdt = make_float3(0.f);
		lut = make_float3(0.f);
		rut = make_float3(0.f);

		rdea = make_float3(d_transforms[6 * (nodeind + 1) + 0], d_transforms[6 * (nodeind + 1) + 1], d_transforms[6 * (nodeind + 1) + 2]);
		rdt = make_float3(d_transforms[6 * (nodeind + 1) + 3], d_transforms[6 * (nodeind + 1) + 4], d_transforms[6 * (nodeind + 1) + 5]);
		luea = make_float3(d_transforms[6 * (nodeind + nodeW) + 0], d_transforms[6 * (nodeind + nodeW) + 1], d_transforms[6 * (nodeind + nodeW) + 2]);
		lut = make_float3(d_transforms[6 * (nodeind + nodeW) + 3], d_transforms[6 * (nodeind + nodeW) + 4], d_transforms[6 * (nodeind + nodeW) + 5]);
		ruea = make_float3(d_transforms[6 * (nodeind + nodeW + 1) + 0], d_transforms[6 * (nodeind + nodeW + 1) + 1], d_transforms[6 * (nodeind + nodeW + 1) + 2]);
		rut = make_float3(d_transforms[6 * (nodeind + nodeW + 1) + 3], d_transforms[6 * (nodeind + nodeW + 1) + 4], d_transforms[6 * (nodeind + nodeW + 1) + 5]);


		float3 resea, rest;
		resea = (1.f - weight.x) * (1.f - weight.y) *ldea +
			weight.x * (1.f - weight.y) * rdea +
			(1.f - weight.x) * weight.y * luea +
			weight.x * weight.y * ruea;
		rest = (1.f - weight.x) * (1.f - weight.y) *ldt +
			weight.x * (1.f - weight.y) * rdt +
			(1.f - weight.x) * weight.y * lut +
			weight.x * weight.y * rut;

		float3x3 rot = evalRMat(resea);
		float4x4 transform = float4x4(rot);
		transform(0, 3) = rest.x;
		transform(1, 3) = rest.y;
		transform(2, 3) = rest.z;
		transform(3, 3) = 1;

		output[x] = MINF;

		if (pInput.x != MINF && !nInput.x != MINF && iInput != MINF)
		{

			//
			index.x;
			index.y;
			//

			//mat3x3 I = intrinsics;
			float4 pInputTransformed = transform * make_float4(pInput, 1);
			float3 pProjTrans = intrinsics * make_float3(pInputTransformed);

			if (pProjTrans.z > 0.f) {

				float2 uvModel = dehomogenize(pProjTrans);

				float3 iTargetAndDerivative = make_float3(bilinearInterpolationFloat4(uvModel.x, uvModel.y, cameraTrackingInput.d_targetIntensityAndDerivatives, imageWidth, imageHeight));

				float iTarget = iTargetAndDerivative.x; // Intensity, uv gradient

				output[x] = iTarget;
			}
		}
	}
}

extern "C" void renderCorrespondenceLocalCUDA(unsigned int imageWidth, unsigned int imageHeight, float *output, CameraTrackingLocalInput cameraTrackingInput, float* intrinsics, CameraTrackingLocalParameters cameraTrackingIParameters, float* d_transforms) {

	const int threadPerBlock = 64;

	//	dim3  block(threadPerBlock, threadPerBlock);
	//	dim3 grid((imageWidth + threadPerBlock - 1) / threadPerBlock, (imageHeight + threadPerBlock - 1) / threadPerBlock);

	dim3 block(threadPerBlock);
	dim3 grid((imageWidth * imageHeight + threadPerBlock - 1) / threadPerBlock);

	renderCorrespondenceLocal_kernel << < grid, block >> > (imageWidth, imageHeight, output, cameraTrackingInput, float3x3(intrinsics), cameraTrackingIParameters, d_transforms);

}

__global__ void renderCorrespondenceLocal2_kernel(unsigned int imageWidth, unsigned int imageHeight, float *output, CameraTrackingLocalInput cameraTrackingInput, float3x3 intrinsics, CameraTrackingLocalParameters cameraTrackingParameters, float3 *d_x_rot, float3 *d_x_trans) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	const int index1D = x;
	const uint2 index = make_uint2(x % imageWidth, x / imageWidth);
	const int nodeW = cameraTrackingParameters.nodeWidth;
	const int nodeH = cameraTrackingParameters.nodeHeight;

	cameraTrackingParameters.offset;

	if (index.x < imageWidth && index.y < imageHeight) {

		float3 pInput = make_float3(cameraTrackingInput.d_inputPos[index1D]);
		float3 nInput = make_float3(cameraTrackingInput.d_inputNormal[index1D]);
		float iInput = cameraTrackingInput.d_inputIntensity[index1D];

		float2 posLocal = (make_float2(index) - cameraTrackingParameters.offset) / cameraTrackingParameters.cellWH;

		if (posLocal.x > nodeW - 1) posLocal.x = nodeW - 1.000001;
		if (posLocal.x < 0) posLocal.x = 0.000001;
		if (posLocal.y > nodeH - 1) posLocal.y = nodeH - 1.000001;
		if (posLocal.y < 0) posLocal.y = 0.000001;

		int2 posi = make_int2(posLocal);
		int nodeind = posi.y * nodeW + posi.x;

		float2 weight = make_float2(posLocal.x - (int)posLocal.x, posLocal.y - (int)posLocal.y);

	
		float3 ldea, rdea, luea, ruea;
		float3 ldt, rdt, lut, rut;

		ldea = d_x_rot[nodeind];
		ldt =d_x_trans[nodeind];

		rdea = make_float3(0.f);
		luea = make_float3(0.f);
		ruea = make_float3(0.f);
		rdt = make_float3(0.f);
		lut = make_float3(0.f);
		rut = make_float3(0.f);

		if (posi.x < nodeW - 1) {
			rdea = d_x_rot[nodeind+1];
			rdt = d_x_trans[nodeind+1];
		}

		if (posi.y < nodeH - 1) {
			luea = d_x_rot[nodeind + nodeW];
			lut = d_x_trans[nodeind + nodeW];
		}
		if (posi.x < nodeW - 1.f && posi.y < nodeH - 1) {
			ruea = d_x_rot[nodeind + nodeW + 1];
			rut = d_x_trans[nodeind + nodeW +  1];
		}


		float3 resea, rest;
		resea = (1.f - weight.x) * (1.f - weight.y) *ldea +
			weight.x * (1.f - weight.y) * rdea +
			(1.f - weight.x) * weight.y * luea +
			weight.x * weight.y * ruea;
		rest = (1.f - weight.x) * (1.f - weight.y) *ldt +
			weight.x * (1.f - weight.y) * rdt +
			(1.f - weight.x) * weight.y * lut +
			weight.x * weight.y * rut;

		float3x3 rot = evalRMat(resea);
		float4x4 transform = float4x4(rot);
		transform(0, 3) = rest.x;
		transform(1, 3) = rest.y;
		transform(2, 3) = rest.z;
		transform(3, 3) = 1;

		output[x] = MINF;

		if (pInput.x != MINF && !nInput.x != MINF && iInput != MINF)
		{
			
			//mat3x3 I = intrinsics;
			float4 pInputTransformed = transform * make_float4(pInput, 1);
			float3 pProjTrans = intrinsics * make_float3(pInputTransformed);

			if (pProjTrans.z > 0.f) {

				float2 uvModel = dehomogenize(pProjTrans);

				float3 iTargetAndDerivative = make_float3(bilinearInterpolationFloat4(uvModel.x, uvModel.y, cameraTrackingInput.d_targetIntensityAndDerivatives, imageWidth, imageHeight));

				float iTarget = iTargetAndDerivative.x; // Intensity, uv gradient

				output[x] = iTarget;
			}
		}
	}
}


extern "C" void renderCorrespondenceLocal2CUDA(unsigned int imageWidth, unsigned int imageHeight, float *output, CameraTrackingLocalInput cameraTrackingInput, float* intrinsics, CameraTrackingLocalParameters cameraTrackingIParameters, float3* d_x_rot, float3* d_x_trans) {

	const int threadPerBlock = 64;

	//	dim3  block(threadPerBlock, threadPerBlock);
	//	dim3 grid((imageWidth + threadPerBlock - 1) / threadPerBlock, (imageHeight + threadPerBlock - 1) / threadPerBlock);

	dim3 block(threadPerBlock);
	dim3 grid((imageWidth * imageHeight + threadPerBlock - 1) / threadPerBlock);

	renderCorrespondenceLocal2_kernel <<<grid, block>>> (imageWidth, imageHeight, output, cameraTrackingInput, float3x3(intrinsics), cameraTrackingIParameters, d_x_rot, d_x_trans);

}


__global__ void setTransform_kernel(float3 *d_xOld_rot, float3 *d_xOld_trans, float3 init_rot, float3 init_trans, int nodeN) {

	int ind = blockIdx.x * blockDim.x + threadIdx.x;

	if (ind < nodeN) {
		d_xOld_rot[ind] = init_rot;
		d_xOld_trans[ind] = init_trans;

	}
}

extern "C" void setTransform(float3 *d_xold_rot, float3 *d_xold_trans, float3 initrot, float3 inittrans, CameraTrackingLocalParameters cameraTrackingParameters) {

	const int nodeN = cameraTrackingParameters.nodeWidth * cameraTrackingParameters.nodeHeight;
	const int blockNodeN = (nodeN + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

	setTransform_kernel <<<blockNodeN, THREAD_PER_BLOCK >>> (d_xold_rot, d_xold_trans, initrot, inittrans, nodeN);

}

__global__ void upsampleGrid_kernel(float3 *d_xold_rot, float3 *d_xold_trans, float3 *d_x_rot, float3 *d_x_trans, float2 offsetCur, float2 offset, float2 cellWHCur, float2 cellWH, int nodeWCur, int nodeW, int nodeHCur, int nodeH) {

	int nodeNCur = nodeWCur * nodeHCur;
	int nodeN = nodeW * nodeH;

	int ind = blockIdx.x * blockDim.x + threadIdx.x;

	if (ind < nodeNCur) {

		int nodeX = ind % nodeWCur;
		int nodeY = ind / nodeWCur;


		//compute the point on prev space
		float2 posCur = offsetCur + make_float2(cellWHCur.x * nodeX, cellWHCur.y*nodeY);

		posCur *= 0.5;// multiplied by factor
		float2 posCurLocal = (posCur - offset) / cellWH;

		if (posCurLocal.x > nodeW - 1) posCurLocal.x = nodeW - 1.000001;
		if (posCurLocal.x < 0) posCurLocal.x = 0.000001;
		if (posCurLocal.y > nodeH - 1) posCurLocal.y = nodeH - 1.000001;
		if (posCurLocal.y < 0) posCurLocal.y = 0.000001;

		int2 posi = make_int2(posCurLocal);
		int index = posi.y * nodeW + posi.x;

		float2 weight = make_float2(posCurLocal.x - (int)posCurLocal.x, posCurLocal.y - (int)posCurLocal.y);

		float3 ld_rot, rd_rot, lu_rot, ru_rot;
		float3 ld_trans, rd_trans, lu_trans, ru_trans;
		float val = 0.f;

		ld_rot = make_float3(0.f);
		lu_rot = make_float3(0.f);
		rd_rot = make_float3(0.f);
		ru_rot = make_float3(0.f);
		ld_trans = make_float3(0.f);
		lu_trans = make_float3(0.f);
		rd_trans = make_float3(0.f);
		ru_trans = make_float3(0.f);

		ld_rot = d_x_rot[index];
		ld_trans = d_x_trans[index];

		val += (1.f - weight.x) * (1.f - weight.y);

		if (posi.x < nodeW - 1) {
			rd_rot = d_x_rot[index+1];
			rd_trans = d_x_trans[index+1];
			val += weight.x * (1.f - weight.y);
		}
		if (posi.y < nodeH - 1) {
//			lu_rot = d_x_rot[index + nodeW];
//			lu_trans = d_x_trans[index + nodeW];
//		}
//		if (posi.x < nodeW - 1 && posi.y < nodeH - 1) {
//			ru_rot = d_x_rot[index + nodeW + 1];
//			ru_trans = d_x_trans[index + nodeW + 1];
			lu_rot = d_x_rot[index+ nodeW];
			lu_trans = d_x_trans[index+ nodeW];
			val += (1.f - weight.x) * (weight.y);
		}
		if (posi.x < nodeW - 1.f && posi.y < nodeH - 1.f ) {
			ru_rot = d_x_rot[index+nodeW+ 1];
			ru_trans = d_x_trans[index+nodeW + 1];
			val += (weight.x) * (weight.y);

		}

		float3 resRot = (1.f - weight.x) * (1.f - weight.y) *ld_rot +
			weight.x * (1.f - weight.y) *rd_rot +
			(1.f - weight.x) * weight.y * lu_rot +
			weight.x * weight.y * ru_rot;
		float3 resTrans = (1.f - weight.x) * (1.f - weight.y) *ld_trans +
			weight.x * (1.f - weight.y) *rd_trans +
			(1.f - weight.x) * weight.y * lu_trans +
			weight.x * weight.y * ru_trans;

		resRot /= val;
		resTrans /= val;
		d_xold_rot[ind] = resRot;
		d_xold_trans[ind] = resTrans;

	}
}

extern "C" void upsampleGrid(float3 *d_xold_rot, float3 *d_xold_trans, float3 *d_x_rot, float3 *d_x_trans, float2 offsetCur, float2 offset, float2 cellWHCur, float2 cellWH, int nodeWCur, int nodeW, int nodeHCur, int nodeH) {

	const int nodeNCur = nodeWCur * nodeHCur;
	const int blockNodeN = (nodeNCur + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

	upsampleGrid_kernel <<<blockNodeN, THREAD_PER_BLOCK >>> (d_xold_rot, d_xold_trans, d_x_rot, d_x_trans, offsetCur, offset, cellWHCur, cellWH, nodeWCur, nodeW, nodeHCur, nodeH);

}



__global__ void updateTransforms_kernel(float3 *d_x_rot, float3 *d_x_trans,  float3 *d_xOld_rot, float3 *d_xOld_trans, float3 *d_xStep_rot, float3 *d_xStep_trans, int nodeN) {


	int ind = blockIdx.x * blockDim.x + threadIdx.x;

	if (0 <= ind && ind < nodeN) {

		d_x_rot[ind] = d_xOld_rot[ind] + d_xStep_rot[ind];
		d_x_trans[ind] = d_xOld_trans[ind] + d_xStep_trans[ind];

	}
}

extern "C" void updateTransforms(float3 *d_x_rot, float3 *d_x_trans, float3 *d_xOld_rot, float3 *d_xOld_trans, float3 *d_xStep_rot, float3 *d_xStep_trans, int nodeN) {

	const int blockNodeN = (nodeN + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

	updateTransforms_kernel <<<blockNodeN, THREAD_PER_BLOCK >>> (d_x_rot, d_x_trans, d_xOld_rot, d_xOld_trans, d_xStep_rot, d_xStep_trans, nodeN);

}

__global__ void renderMotionMapKernel( float *d_x_map, float3 *d_x_rot, float3 *d_x_trans, float2 offset, float2 cellWH, int imageWidth, int imageHeight, int nodeWidth, int nodeHeight) {

	int pixelx = threadIdx.x + blockIdx.x * blockDim.x;
	int pixely = threadIdx.y + blockIdx.y * blockDim.y;

	if (pixelx < imageWidth && pixely < imageHeight) {

		int pixelIndex = pixely * imageWidth + pixelx;
		float2 pixelxyf = make_float2(pixelx, pixely);

		float2 cellxy = (pixelxyf - offset) / cellWH;
		
		if (cellxy.x > nodeWidth - 1.f)
			cellxy.x = nodeWidth - 1.001f;
		if (cellxy.y > nodeHeight - 1.f)
			cellxy.y = nodeHeight - 1.001f;
		if (cellxy.x < 0.f)
			cellxy.x = 0.001f;
		if (cellxy.y < 0.f)
			cellxy.y = 0.001f;
		
		int2 cellxyi = make_int2(cellxy);
		int cellIdx1D = cellxyi.y * nodeWidth + cellxyi.x;
		float2 weight = cellxy - make_float2(cellxyi);
		float3 resrot, restrans;

		resrot = make_float3(0.f);
		restrans = make_float3(0.f);
	
		//ld
		resrot = (1.f - weight.x)* (1.f - weight.y)* d_x_rot[cellIdx1D]
		+ weight.x* (1.f - weight.y)* d_x_rot[cellIdx1D + 1]
		+ (1.f - weight.x)* weight.y * d_x_rot[cellIdx1D + nodeWidth]
		+ weight.x * weight.y * d_x_rot[cellIdx1D + nodeWidth + 1];
		restrans = (1.f - weight.x)* (1.f - weight.y)* d_x_trans[cellIdx1D]
			+ weight.x* (1.f - weight.y)* d_x_trans[cellIdx1D + 1]
			+ (1.f - weight.x)* weight.y * d_x_trans[cellIdx1D + nodeWidth]
			+ weight.x * weight.y * d_x_trans[cellIdx1D + nodeWidth + 1];

		d_x_map[6 * pixelIndex + 0] = resrot.x;
		d_x_map[6 * pixelIndex + 1] = resrot.y;
		d_x_map[6 * pixelIndex + 2] = resrot.z;
		d_x_map[6 * pixelIndex + 3] = restrans.x;
		d_x_map[6 * pixelIndex + 4] = restrans.y;
		d_x_map[6 * pixelIndex + 5] = restrans.z;

	}

}

extern"C" void renderMotionMapCUDA(float * d_x_map, float3 *d_x_rot, float3 *d_x_trans, float2 offset, float2 cellWH, int imageWidth, int imageHeight, int nodeWidth, int nodeHeight) {

	const dim3 block (THREAD_PER_BLOCK, THREAD_PER_BLOCK);
	const dim3 grid((imageWidth + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, (imageHeight + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);

	renderMotionMapKernel <<<grid, block >>> (d_x_map, d_x_rot,d_x_trans, offset, cellWH, imageWidth, imageHeight, nodeWidth, nodeHeight);

}