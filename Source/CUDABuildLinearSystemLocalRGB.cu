#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"
#include "ICPUtil.h"
#include "CameraTrackingInput.h"

/////////////////////////////////////////////////////
// Defines
/////////////////////////////////////////////////////

#define _DEBUG

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE 30
#endif

#define MINF __int_as_float(0xff800000)

/////////////////////////////////////////////////////
// Shared Memory
/////////////////////////////////////////////////////

__shared__ float bucket4[ARRAY_SIZE*BLOCK_SIZE];

/////////////////////////////////////////////////////
// Helper Functions
/////////////////////////////////////////////////////

__device__ inline void addToLocalScanElement(uint inpGTid, uint resGTid, volatile float* shared)
{
#pragma unroll
	for (uint i = 0; i<ARRAY_SIZE; i++)
	{
		shared[ARRAY_SIZE*resGTid + i] += shared[ARRAY_SIZE*inpGTid + i];
	}
}

__device__ inline void CopyToResultScanElement(uint GID, float* output)
{
#pragma unroll
	for (uint i = 0; i<ARRAY_SIZE; i++)
	{
		output[ARRAY_SIZE*GID + i] = bucket4[0 + i];
	}
}

__device__ inline void CopyToResultScanElementGlobal(uint GID, float* output, float *d_gtemp)
{
#pragma unroll
	for (uint i = 0; i<ARRAY_SIZE; i++)
	{
		output[ARRAY_SIZE*GID + i] = d_gtemp[GID * ARRAY_SIZE * BLOCK_SIZE + i];
	}
}

__device__ inline void CopyToResultScanElementGlobal(uint GID, float* output, float *d_gtemp, float lambda, float3 &stepRot, float3 &stepTrans)
{

#pragma unroll
	for (uint i = 0; i < ARRAY_SIZE; i++)
	{
		output[ARRAY_SIZE*GID + i] = d_gtemp[GID * ARRAY_SIZE * BLOCK_SIZE + i];
	}

	output[ARRAY_SIZE*GID + 0] += lambda;//6
	output[ARRAY_SIZE*GID + 6] += lambda;//5  = 11
	output[ARRAY_SIZE*GID + 11] += lambda;//4 = 15
	output[ARRAY_SIZE*GID + 15] += lambda;//3 = 18
	output[ARRAY_SIZE*GID + 18] += lambda;//2 = 20
	output[ARRAY_SIZE*GID + 20] += lambda;//1
	output[ARRAY_SIZE*GID + 21] -= lambda * stepRot.x;
	output[ARRAY_SIZE*GID + 22] -= lambda * stepRot.y;
	output[ARRAY_SIZE*GID + 23] -= lambda * stepRot.z;
	output[ARRAY_SIZE*GID + 24] -= lambda * stepTrans.x;
	output[ARRAY_SIZE*GID + 25] -= lambda * stepTrans.y;
	output[ARRAY_SIZE*GID + 26] -= lambda * stepTrans.z;
	output[ARRAY_SIZE*GID + 27] += lambda * (dot ( stepRot, stepRot ) + dot ( stepTrans, stepTrans ));
	output[ARRAY_SIZE*GID + 28] += lambda;
	output[ARRAY_SIZE*GID + 29] += 1;

}


__device__ inline void addToLocalScanElementGlobal(uint nodeInd, uint inpGTid, uint resGTid, float* d_gtemp)
{
#pragma unroll
	for (uint i = 0; i<ARRAY_SIZE; i++)
	{
		d_gtemp[nodeInd* ARRAY_SIZE * BLOCK_SIZE + ARRAY_SIZE*resGTid + i] += d_gtemp[nodeInd* ARRAY_SIZE * BLOCK_SIZE + ARRAY_SIZE*inpGTid + i];
	}
}


__device__ inline void SetZeroScanElement(uint GTid)
{
#pragma unroll
	for (uint i = 0; i<ARRAY_SIZE; i++)
	{
		bucket4[GTid*ARRAY_SIZE + i] = 0.0f;
	}
}

__device__ inline float weightDist(float2 src, float2 tar, float sigma) {
	float2 diff = src - tar;

	return exp (-( diff.x * diff.x + diff.y * diff.y ) / (2 *sigma*sigma));

}

/////////////////////////////////////////////////////
// Scan
/////////////////////////////////////////////////////

__device__ inline void warpReduce(int GTid) // See Optimizing Parallel Reduction in CUDA by Mark Harris
{
	addToLocalScanElement(GTid + 32, GTid, bucket4);
	addToLocalScanElement(GTid + 16, GTid, bucket4);
	addToLocalScanElement(GTid + 8, GTid, bucket4);
	addToLocalScanElement(GTid + 4, GTid, bucket4);
	addToLocalScanElement(GTid + 2, GTid, bucket4);
	addToLocalScanElement(GTid + 1, GTid, bucket4);
}

__device__ inline void warpReduceGlobal(int nodeInd, int GTid, float* d_gtemp) // See Optimizing Parallel Reduction in CUDA by Mark Harris
{
	addToLocalScanElementGlobal(nodeInd, GTid + 32, GTid, d_gtemp);
//	__syncthread();
	addToLocalScanElementGlobal(nodeInd, GTid + 16, GTid, d_gtemp);
	addToLocalScanElementGlobal(nodeInd, GTid + 8, GTid, d_gtemp);
	addToLocalScanElementGlobal(nodeInd, GTid + 4, GTid, d_gtemp);
	addToLocalScanElementGlobal(nodeInd, GTid + 2, GTid, d_gtemp);
	addToLocalScanElementGlobal(nodeInd, GTid + 1, GTid, d_gtemp);
}

/////////////////////////////////////////////////////
// Compute Normal Equations
/////////////////////////////////////////////////////

__device__ inline  void addToLocalSystemGlobal(mat1x6& jacobianBlockRow, mat1x1& residualsBlockRow, float weight, uint blockInd, uint threadIdx, float* d_gtemp)
{
	uint linRowStart = 0;

#pragma unroll
	for (uint i = 0; i<6; i++)
	{

		mat1x1 colI; jacobianBlockRow.getBlock(0, i, colI);

#pragma unroll
		for (uint j = i; j<6; j++)
		{
			mat1x1 colJ; jacobianBlockRow.getBlock(0, j, colJ);

			d_gtemp [blockInd * ARRAY_SIZE* BLOCK_SIZE + ARRAY_SIZE*threadIdx + linRowStart + j - i] += colI.getTranspose()*colJ*weight;
		}

		linRowStart += 6 - i;

		d_gtemp [blockInd * ARRAY_SIZE* BLOCK_SIZE + ARRAY_SIZE*threadIdx + 21 + i] -= colI.getTranspose()*residualsBlockRow*weight; // -JTF

	}

	d_gtemp [blockInd * ARRAY_SIZE* BLOCK_SIZE + ARRAY_SIZE*threadIdx + 27] += weight*residualsBlockRow.norm1DSquared(); // residual
	d_gtemp [blockInd * ARRAY_SIZE* BLOCK_SIZE + ARRAY_SIZE*threadIdx + 28] += weight;									 // weight
	d_gtemp [blockInd * ARRAY_SIZE* BLOCK_SIZE + ARRAY_SIZE*threadIdx + 29] += 1.0f;									 // corr number
}


__device__ inline  void addToLocalSystem(mat1x6& jacobianBlockRow, mat1x1& residualsBlockRow, float weight, uint threadIdx,  volatile float* shared)
{
	uint linRowStart = 0;

#pragma unroll
	for (uint i = 0; i<6; i++)
	{
		mat1x1 colI; jacobianBlockRow.getBlock(0, i, colI);

#pragma unroll
		for (uint j = i; j<6; j++)
		{
			mat1x1 colJ; jacobianBlockRow.getBlock(0, j, colJ);

			shared[ARRAY_SIZE*threadIdx + linRowStart + j - i] += colI.getTranspose()*colJ*weight;
		}

		linRowStart += 6 - i;

		shared[ARRAY_SIZE*threadIdx + 21 + i] -= colI.getTranspose()*residualsBlockRow*weight; // -JTF
	}

	shared[ARRAY_SIZE*threadIdx + 27] += weight*residualsBlockRow.norm1DSquared(); // residual
	shared[ARRAY_SIZE*threadIdx + 28] += weight;									 // weight

	shared[ARRAY_SIZE*threadIdx + 29] += 1.0f;									 // corr number
}


__global__ void scanNormalEquationsLocalDevice(unsigned int imageWidth, unsigned int imageHeight, float* output, CameraTrackingLocalInput cameraTrackingInput, float3x3 intrinsics, CameraTrackingLocalParameters cameraTrackingIParameters, float3 anglesOld, float3 translationOld, float2 pos, unsigned int localWindowSize)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int localWindowWidth = 2 * cameraTrackingIParameters.localWindowHWidth + 1;
	int2 posi = make_int2(pos.x - cameraTrackingIParameters.localWindowHWidth, pos.y - cameraTrackingIParameters.localWindowHWidth);

	// Set system to zero
	SetZeroScanElement(threadIdx.x);

	//Locally sum small window
#pragma unroll
	for (uint i = 0; i < localWindowSize; i++)
	{
		const int localIndex1D = localWindowSize*x + i;
		int2 index = make_int2(localIndex1D%localWindowWidth, localIndex1D / localWindowWidth);

		index += posi;
		const int index1D = index.x + index.y * imageWidth;


		if (0 <= index.x && index.x < imageWidth && 0 <= index.y && index.y < imageHeight)
		{
			mat3x1 pInput = mat3x1(make_float3(cameraTrackingInput.d_inputPos[index1D]));
			mat3x1 nInput = mat3x1(make_float3(cameraTrackingInput.d_inputNormal[index1D]));		
			mat1x1 iInput = mat1x1(cameraTrackingInput.d_inputIntensity[index1D]);


			if (!pInput.checkMINF() && !nInput.checkMINF() && !iInput.checkMINF()&& iInput>1.f/255.f)
			{
				mat3x3 I = mat3x3(intrinsics);
				mat3x3 ROld = mat3x3(evalRMat(anglesOld));
				mat3x1 pInputTransformed = ROld*pInput + mat3x1(translationOld);
				mat3x1 nInputTransformed = ROld*nInput;

				mat3x3 Ralpha = mat3x3(evalR_dGamma(anglesOld));
				mat3x3 Rbeta = mat3x3(evalR_dBeta(anglesOld));
				mat3x3 Rgamma = mat3x3(evalR_dAlpha(anglesOld));

				mat3x1 pProjTrans = I*pInputTransformed;

				if (pProjTrans(2) > 0.0f)
				{
					mat2x1 uvModel = mat2x1(dehomogenize(pProjTrans));

					mat3x1 iTargetAndDerivative = mat3x1(make_float3(bilinearInterpolationFloat4(uvModel(0), uvModel(1), cameraTrackingInput.d_targetIntensityAndDerivatives, imageWidth, imageHeight)));
					mat1x1 iTarget = mat1x1(iTargetAndDerivative(0)); mat1x1 iTargetDerivUIntensity = mat1x1(iTargetAndDerivative(1)); mat1x1 iTargetDerivVIntensity = mat1x1(iTargetAndDerivative(2));

					mat2x3 PI = dehomogenizeDerivative(pProjTrans);
					mat3x1 phiAlpha = Ralpha*pInputTransformed; mat3x1 phiBeta = Rbeta *pInputTransformed;	mat3x1 phiGamma = Rgamma*pInputTransformed;

					if (!iTarget.checkMINF() && !iTargetDerivUIntensity.checkMINF() && !iTargetDerivVIntensity.checkMINF())
					{

						// Color
						mat1x1 diffIntensity(iTarget - iInput);
						mat1x2 DIntensity; DIntensity(0, 0) = iTargetDerivUIntensity(0); DIntensity(0, 1) = iTargetDerivVIntensity(0);
						if (DIntensity.norm1D() > cameraTrackingIParameters.colorGradiantMin)
						{
							float weightColor = max(0.0f, 1.0f - diffIntensity.norm1D() / cameraTrackingIParameters.colorThres);

							weightColor = weightDist(make_float2(index.x, index.y), pos, cameraTrackingIParameters.sigma);

							mat1x3 tmp0Intensity = DIntensity*PI*I;

							mat1x6 jacobianBlockRowIntensity;
							jacobianBlockRowIntensity.setBlock(tmp0Intensity*phiAlpha, 0, 0);
							jacobianBlockRowIntensity.setBlock(tmp0Intensity*phiBeta, 0, 1);
							jacobianBlockRowIntensity.setBlock(tmp0Intensity*phiGamma, 0, 2);
							jacobianBlockRowIntensity.setBlock(tmp0Intensity, 0, 3);
							addToLocalSystem(jacobianBlockRowIntensity, diffIntensity, weightColor, threadIdx.x, bucket4);
						}
					}
				}
			}
		}
	}

	__syncthreads();

	// Up sweep 2D
#pragma unroll
	for (unsigned int stride = BLOCK_SIZE / 2; stride > 32; stride >>= 1)
	{
		if (threadIdx.x < stride) addToLocalScanElement(threadIdx.x + stride / 2, threadIdx.x, bucket4);

		__syncthreads();
	}

	if (threadIdx.x < 32) warpReduce(threadIdx.x);

	// Copy to output texture
	if (threadIdx.x == 0) CopyToResultScanElement(blockIdx.x, output);
}

__global__ void scanNormalEquationLocalDevice() {

}

__global__ void scanNormalEquationsLocalParallelDevice(unsigned int imageWidth, unsigned int imageHeight, float* d_system,float *d_temp, float3* d_x_rot, float3* d_x_trans, CameraTrackingLocalInput cameraTrackingInput, float3x3 intrinsics, CameraTrackingLocalParameters cameraTrackingParameters)
{

	unsigned int nodeInd = blockIdx.x;
	unsigned int threadInd = threadIdx.x;

	const int nodeW = cameraTrackingParameters.nodeWidth;
	const int nodeH = cameraTrackingParameters.nodeHeight;
	const int nodeIndX = nodeInd % nodeW;
	const int nodeIndY = nodeInd / nodeW;
	const int imageW = cameraTrackingParameters.imageWidth;
	const int imageH = cameraTrackingParameters.imageHeight;
	const int localWindowHWidth = cameraTrackingParameters.localWindowHWidth;
	const float2 cellWH = cameraTrackingParameters.cellWH;

	float2 nodePos = cellWH * make_float2(nodeIndX, nodeIndY) + cameraTrackingParameters.offset;
	int2 nodePosi = make_int2(nodePos);

	int six = max(0, nodePosi.x - localWindowHWidth);
	int eix = min(imageW - 1, nodePosi.x + localWindowHWidth + 1);
	int siy = max(0, nodePosi.y - localWindowHWidth);
	int eiy = min(imageH - 1, nodePosi.y + localWindowHWidth + 1);

	int rangeW = eix - six; // sx <= x < ex
	int rangeH = eiy - siy; // sy <= y < ey
	int rangeN = rangeW * rangeH;

	for (int i = threadInd; i < rangeN; i += BLOCK_SIZE) {
		
		int offix = i % rangeW;
		int offiy = i / rangeW;
		int ix = six + offix;
		int iy = siy + offiy;
		int index1D = iy * imageW + ix;

		mat3x1 pInput = mat3x1(make_float3(cameraTrackingInput.d_inputPos[index1D]));
		mat3x1 nInput = mat3x1(make_float3(cameraTrackingInput.d_inputNormal[index1D]));
		mat1x1 iInput = mat1x1(cameraTrackingInput.d_inputIntensity[index1D]);
		float maskVal = cameraTrackingInput.d_inputMask[index1D];

		if (!pInput.checkMINF() && !nInput.checkMINF() && !iInput.checkMINF() && maskVal < 0.5 && iInput > 1.f / 255.f)
		{
			//
			float3 xRot = d_x_rot[nodeInd];
			float3 xTrans = d_x_trans[nodeInd];

			float3 aRot, aTrans;

			mat3x3 I = mat3x3(intrinsics);
			mat3x3 ROld = mat3x3(evalRMat(xRot));
			mat3x1 pInputTransformed = ROld*pInput + mat3x1(xTrans);
			mat3x1 nInputTransformed = ROld*nInput;

			mat3x3 Ralpha = mat3x3(evalR_dGamma(xRot));
			mat3x3 Rbeta = mat3x3(evalR_dBeta(xRot));
			mat3x3 Rgamma = mat3x3(evalR_dAlpha(xRot));

			mat3x1 pProjTrans = I*pInputTransformed;

			if (pProjTrans(2) > 0.0f)
			{
				mat2x1 uvModel = mat2x1(dehomogenize(pProjTrans));

				mat3x1 iTargetAndDerivative = mat3x1(make_float3(bilinearInterpolationFloat4(uvModel(0), uvModel(1), cameraTrackingInput.d_targetIntensityAndDerivatives, imageW, imageH)));
				mat1x1 iTarget = mat1x1(iTargetAndDerivative(0)); mat1x1 iTargetDerivUIntensity = mat1x1(iTargetAndDerivative(1)); mat1x1 iTargetDerivVIntensity = mat1x1(iTargetAndDerivative(2));

				mat2x3 PI = dehomogenizeDerivative(pProjTrans);
				mat3x1 phiAlpha = Ralpha*pInputTransformed; mat3x1 phiBeta = Rbeta *pInputTransformed;	mat3x1 phiGamma = Rgamma*pInputTransformed;

				if (!iTarget.checkMINF() && !iTargetDerivUIntensity.checkMINF() && !iTargetDerivVIntensity.checkMINF())
				{

					// Color
					mat1x1 diffIntensity(iTarget - iInput);
					mat1x2 DIntensity; DIntensity(0, 0) = iTargetDerivUIntensity(0); DIntensity(0, 1) = iTargetDerivVIntensity(0);


					if (DIntensity.norm1D() > cameraTrackingParameters.colorGradiantMin)
					{
						float weightColor;
						//= max(0.0f, 1.0f - diffIntensity.norm1D() / cameraTrackingParameters.colorThres);

						weightColor = weightDist(make_float2(ix, iy), nodePos, cameraTrackingParameters.sigma);
						mat1x3 tmp0Intensity = DIntensity*PI*I;

						mat1x6 jacobianBlockRowIntensity;
						jacobianBlockRowIntensity.setBlock(tmp0Intensity*phiAlpha, 0, 0);
						jacobianBlockRowIntensity.setBlock(tmp0Intensity*phiBeta, 0, 1);
						jacobianBlockRowIntensity.setBlock(tmp0Intensity*phiGamma, 0, 2);
						jacobianBlockRowIntensity.setBlock(tmp0Intensity, 0, 3);
						addToLocalSystemGlobal(jacobianBlockRowIntensity, diffIntensity, weightColor, nodeInd, threadIdx.x, d_temp);

					}
				}
			}
		}
	}

	__syncthreads();

	
	// Up sweep 2D
#pragma unroll
	for (unsigned int stride = BLOCK_SIZE / 2; stride > 32; stride >>= 1)
	{
		if (threadIdx.x < stride) addToLocalScanElementGlobal(nodeInd, threadIdx.x + stride / 2, threadIdx.x, d_temp);

		__syncthreads();

	}
	if (threadIdx.x < 32) warpReduceGlobal(nodeInd, threadIdx.x, d_temp);


//
	__syncthreads();

	// Copy to output texture
	if (threadIdx.x == 0) CopyToResultScanElementGlobal(nodeInd, d_system, d_temp );
}

__global__ void scanNormalEquationsLocalParallelRegDevice(unsigned int imageWidth, unsigned int imageHeight, float* d_system, float *d_temp, float3* d_x_rot, float3* d_x_trans, float3* d_x_step_rot, float3* d_x_step_trans, float lambda, CameraTrackingLocalInput cameraTrackingInput, float3x3 intrinsics, CameraTrackingLocalParameters cameraTrackingParameters)
{

	unsigned int nodeInd = blockIdx.x;
	unsigned int threadInd = threadIdx.x;

	const int nodeW = cameraTrackingParameters.nodeWidth;
	const int nodeH = cameraTrackingParameters.nodeHeight;
	const int nodeIndX = nodeInd % nodeW;
	const int nodeIndY = nodeInd / nodeW;
	const int imageW = cameraTrackingParameters.imageWidth;
	const int imageH = cameraTrackingParameters.imageHeight;
	const int localWindowHWidth = cameraTrackingParameters.localWindowHWidth;
	const float2 cellWH = cameraTrackingParameters.cellWH;

	float2 nodePos = cellWH * make_float2(nodeIndX, nodeIndY) + cameraTrackingParameters.offset;
	int2 nodePosi = make_int2(nodePos);

	int six = max(0, nodePosi.x - localWindowHWidth);
	int eix = min(imageW - 1, nodePosi.x + localWindowHWidth + 1);
	int siy = max(0, nodePosi.y - localWindowHWidth);
	int eiy = min(imageH - 1, nodePosi.y + localWindowHWidth + 1);

	int rangeW = eix - six; // sx <= x < ex
	int rangeH = eiy - siy; // sy <= y < ey
	int rangeN = rangeW * rangeH;

	float3 xRot = d_x_rot[nodeInd] + d_x_step_rot[nodeInd];
	float3 xTrans = d_x_trans[nodeInd] + d_x_step_trans[nodeInd];

	for (int i = threadInd; i < rangeN; i += BLOCK_SIZE) {

		int offix = i % rangeW;
		int offiy = i / rangeW;
		int ix = six + offix;
		int iy = siy + offiy;
		int index1D = iy * imageW + ix;

		mat3x1 pInput = mat3x1(make_float3(cameraTrackingInput.d_inputPos[index1D]));
		mat3x1 nInput = mat3x1(make_float3(cameraTrackingInput.d_inputNormal[index1D]));
		mat1x1 iInput = mat1x1(cameraTrackingInput.d_inputIntensity[index1D]);
		float maskVal = cameraTrackingInput.d_inputMask[index1D];

		if (!pInput.checkMINF() && !nInput.checkMINF() && !iInput.checkMINF() && maskVal < 0.5 && iInput > 1.f / 255.f)
		{
			//


			float3 aRot, aTrans;

			mat3x3 I = mat3x3(intrinsics);
			mat3x3 ROld = mat3x3(evalRMat(xRot));
			mat3x1 pInputTransformed = ROld*pInput + mat3x1(xTrans);
			mat3x1 nInputTransformed = ROld*nInput;

			mat3x3 Ralpha = mat3x3(evalR_dGamma(xRot));
			mat3x3 Rbeta = mat3x3(evalR_dBeta(xRot));
			mat3x3 Rgamma = mat3x3(evalR_dAlpha(xRot));

			mat3x1 pProjTrans = I*pInputTransformed;

			if (pProjTrans(2) > 0.0f)
			{
				mat2x1 uvModel = mat2x1(dehomogenize(pProjTrans));

				mat3x1 iTargetAndDerivative = mat3x1(make_float3(bilinearInterpolationFloat4(uvModel(0), uvModel(1), cameraTrackingInput.d_targetIntensityAndDerivatives, imageW, imageH)));
				mat1x1 iTarget = mat1x1(iTargetAndDerivative(0)); mat1x1 iTargetDerivUIntensity = mat1x1(iTargetAndDerivative(1)); mat1x1 iTargetDerivVIntensity = mat1x1(iTargetAndDerivative(2));

				mat2x3 PI = dehomogenizeDerivative(pProjTrans);
				mat3x1 phiAlpha = Ralpha*pInputTransformed; mat3x1 phiBeta = Rbeta *pInputTransformed;	mat3x1 phiGamma = Rgamma*pInputTransformed;

				if (!iTarget.checkMINF() && !iTargetDerivUIntensity.checkMINF() && !iTargetDerivVIntensity.checkMINF())
				{

					// Color
					mat1x1 diffIntensity(iTarget - iInput);
					mat1x2 DIntensity; DIntensity(0, 0) = iTargetDerivUIntensity(0); DIntensity(0, 1) = iTargetDerivVIntensity(0);


					if (DIntensity.norm1D() > cameraTrackingParameters.colorGradiantMin)
					{
						float weightColor;
						weightColor = weightDist(make_float2(ix, iy), nodePos, cameraTrackingParameters.sigma);
						//*max(0.0f, 1.0f - diffIntensity.norm1D() / cameraTrackingParameters.colorThres);;
						mat1x3 tmp0Intensity = DIntensity*PI*I;

						mat1x6 jacobianBlockRowIntensity;
						jacobianBlockRowIntensity.setBlock(tmp0Intensity*phiAlpha, 0, 0);
						jacobianBlockRowIntensity.setBlock(tmp0Intensity*phiBeta, 0, 1);
						jacobianBlockRowIntensity.setBlock(tmp0Intensity*phiGamma, 0, 2);
						jacobianBlockRowIntensity.setBlock(tmp0Intensity, 0, 3);
						addToLocalSystemGlobal(jacobianBlockRowIntensity, diffIntensity, weightColor, nodeInd, threadIdx.x, d_temp);

					}
				}
			}
		}
	}

	__syncthreads();


	// Up sweep 2D
#pragma unroll
	for (unsigned int stride = BLOCK_SIZE / 2; stride > 32; stride >>= 1)
	{
		if (threadIdx.x < stride) addToLocalScanElementGlobal(nodeInd, threadIdx.x + stride / 2, threadIdx.x, d_temp);

		__syncthreads();

	}
	if (threadIdx.x < 32) warpReduceGlobal(nodeInd, threadIdx.x, d_temp);


	//
	__syncthreads();

	// Copy to output texture
	if (threadIdx.x == 0) {

		CopyToResultScanElementGlobal(nodeInd, d_system, d_temp, lambda, d_x_step_rot[nodeInd], d_x_step_trans[nodeInd]);

	}
}

extern "C" void computeNormalEquationsLocal(unsigned int imageWidth, unsigned int imageHeight, float* output, CameraTrackingLocalInput cameraTrackingInput, float* intrinsics, CameraTrackingLocalParameters cameraTrackingParameters, float3 anglesOld, float3 translationOld, float2 pos, unsigned int localWindowSize, unsigned int blockSizeInt)
{
	const int localWindowWidth = cameraTrackingParameters.localWindowHWidth * 2 + 1;
	const unsigned int numElements = localWindowWidth * localWindowWidth;
	dim3 blockSize(blockSizeInt, 1, 1);
	dim3 gridSize((numElements + blockSizeInt*localWindowSize - 1) / (blockSizeInt*localWindowSize), 1, 1);

	scanNormalEquationsLocalDevice << <gridSize, blockSize >> > (imageWidth, imageHeight, output, cameraTrackingInput, float3x3(intrinsics), cameraTrackingParameters, anglesOld, translationOld, pos, localWindowSize);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


extern "C" void computeNormalEquationsAllLocal(unsigned int imageWidth, unsigned int imageHeight, float* d_system, float *d_temp, float3 *d_x_rot, float3 *d_x_trans, CameraTrackingLocalInput cameraTrackingInput, float* intrinsics, CameraTrackingLocalParameters cameraTrackingParameters){
	

		const int nodeN = cameraTrackingParameters.nodeHeight * cameraTrackingParameters.nodeWidth;


		dim3 blockSize(BLOCK_SIZE, 1, 1);
		dim3 gridSize(nodeN, 1, 1);

		cudaMemset(d_temp, 0,  sizeof(float)* BLOCK_SIZE * ARRAY_SIZE * nodeN);

		scanNormalEquationsLocalParallelDevice << <gridSize, blockSize >> > (imageWidth, imageHeight, d_system, d_temp, d_x_rot, d_x_trans, cameraTrackingInput, float3x3(intrinsics), cameraTrackingParameters);
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
	}

extern "C" void computeNormalEquationsAllRegLocal(unsigned int imageWidth, unsigned int imageHeight, float* d_system, float *d_temp, float3 *d_x_rot, float3 *d_x_trans, float3 *d_x_step_rot, float3 *d_x_step_trans, float lambda, CameraTrackingLocalInput cameraTrackingInput, float* intrinsics, CameraTrackingLocalParameters cameraTrackingParameters) {


	const int nodeN = cameraTrackingParameters.nodeHeight * cameraTrackingParameters.nodeWidth;


	dim3 blockSize(BLOCK_SIZE, 1, 1);
	dim3 gridSize(nodeN, 1, 1);

	cudaMemset(d_temp, 0, sizeof(float)* BLOCK_SIZE * ARRAY_SIZE * nodeN);

	scanNormalEquationsLocalParallelRegDevice << <gridSize, blockSize >> > (imageWidth, imageHeight, d_system, d_temp, d_x_rot, d_x_trans, d_x_step_rot, d_x_step_trans, lambda, cameraTrackingInput, float3x3(intrinsics), cameraTrackingParameters);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}