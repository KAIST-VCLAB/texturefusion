#include "stdafx.h"

#include "CUDABuildLinearSystemLocalRGB.h"

#include "GlobalAppState.h"
#include "GlobalCameraTrackingState.h"

#include "MatrixConversion.h"

#include <iostream>


#define BLOCK_SIZE 256

extern "C" void computeNormalEquationsLocal(unsigned int imageWidth, unsigned int imageHeight, float* output, CameraTrackingLocalInput cameraTrackingInput, float* intrinsics, CameraTrackingLocalParameters cameraTrackingParameters, float3 anglesOld, float3 translationOld, float2 pos, unsigned int localWindowSize, unsigned int blockSize);

extern "C" void computeNormalEquationsAllLocal(unsigned int imageWidth, unsigned int imageHeight, float* d_system, float *d_temp, float3 *d_x_rot, float3 *d_x_trans, CameraTrackingLocalInput cameraTrackingInput, float* intrinsics, CameraTrackingLocalParameters cameraTrackingParameters);

extern "C" void computeNormalEquationsAllRegLocal(unsigned int imageWidth, unsigned int imageHeight, float* d_system, float *d_temp, float3 *d_x_rot, float3 *d_x_trans, float3 *d_x_step_rot, float3 *d_x_step_trans, float lambda, CameraTrackingLocalInput cameraTrackingInput, float* intrinsics, CameraTrackingLocalParameters cameraTrackingParameters);


CUDABuildLinearSystemLocalRGB::CUDABuildLinearSystemLocalRGB(unsigned int imageWidth, unsigned int imageHeight)
{
	cutilSafeCall(cudaMalloc(&d_output, 30 * sizeof(float)*imageWidth*imageHeight));
	h_output = new float[30 * imageWidth*imageHeight];
}

CUDABuildLinearSystemLocalRGB::CUDABuildLinearSystemLocalRGB(unsigned int imageWidth, unsigned int imageHeight, unsigned int nodeWidth, unsigned int nodeHeight)
{
//	cutilSafeCall(cudaMalloc(&d_output, 30 * sizeof(float)*imageWidth*imageHeight));
//	h_output = new float[30 * imageWidth*imageHeight];
	cutilSafeCall(cudaMalloc(&d_temp, 30 * BLOCK_SIZE * sizeof(float)* nodeWidth * nodeHeight));

}

CUDABuildLinearSystemLocalRGB::~CUDABuildLinearSystemLocalRGB() {
	if (d_output) {
		cutilSafeCall(cudaFree(d_output));
	}
	if (h_output) {
		SAFE_DELETE_ARRAY(h_output);
	}
}

void CUDABuildLinearSystemLocalRGB::applyBL(CameraTrackingLocalInput cameraTrackingInput, Eigen::Matrix3f& intrinsics, CameraTrackingLocalParameters cameraTrackingParameters, float3 anglesOld, float3 translationOld, float2 posCenter, unsigned int imageWidth, unsigned int imageHeight, unsigned int level, Matrix6x7f& res, LinearSystemConfidence& conf)
{
	unsigned int localWindowSize = 12;
	unsigned int localWindowWidth = cameraTrackingParameters.localWindowHWidth * 2 + 1;
	if (level != 0) localWindowSize = max(1, localWindowSize / (4 * level));

	const unsigned int blockSize = 64;
	const unsigned int dimX = (unsigned int)ceil(((float)localWindowWidth*localWindowWidth) / (localWindowSize*blockSize));

	Eigen::Matrix3f intrinsicsRowMajor = intrinsics.transpose(); // Eigen is col major / cuda is row major
	computeNormalEquationsLocal(imageWidth, imageHeight, d_output, cameraTrackingInput, intrinsicsRowMajor.data(), cameraTrackingParameters, anglesOld, translationOld, posCenter, localWindowSize, blockSize);

	cutilSafeCall(cudaMemcpy(h_output, d_output, sizeof(float) * 30 * dimX, cudaMemcpyDeviceToHost));

	// Copy to CPU
	res = reductionSystemCPU(h_output, dimX, conf);
}


void CUDABuildLinearSystemLocalRGB::applyBLs(CameraTrackingLocalInput cameraTrackingInput, Eigen::Matrix3f& intrinsics, CameraTrackingLocalParameters cameraTrackingParameters, float3 *d_x_rot, float3 *d_x_trans, float3 *d_x_step_rot, float3 *d_x_step_trans, float lambdaReg, float *d_system, unsigned int level, LinearSystemConfidence& conf)
{

	Eigen::Matrix3f intrinsicsRowMajor = intrinsics.transpose(); // Eigen is col major / cuda is row major
	computeNormalEquationsAllRegLocal(cameraTrackingParameters.imageWidth, cameraTrackingParameters.imageHeight, d_system, d_temp, d_x_rot, d_x_trans, d_x_step_rot, d_x_step_trans, lambdaReg,cameraTrackingInput, intrinsicsRowMajor.data(), cameraTrackingParameters);
	
}


Matrix6x7f CUDABuildLinearSystemLocalRGB::reductionSystemCPU(const float* data, unsigned int nElems, LinearSystemConfidence& conf)
{
	Matrix6x7f res; res.setZero();

	conf.reset();
	float numCorrF = 0.0f;

	for (unsigned int k = 0; k<nElems; k++)
	{
		unsigned int linRowStart = 0;

		for (unsigned int i = 0; i<6; i++)
		{
			for (unsigned int j = i; j<6; j++)
			{
				res(i, j) += data[30 * k + linRowStart + j - i];
			}

			linRowStart += 6 - i;

			res(i, 6) += data[30 * k + 21 + i];
		}

		conf.sumRegError += data[30 * k + 27];
		conf.sumRegWeight += data[30 * k + 28];

		numCorrF += data[30 * k + 29];
	}

	// Fill lower triangle
	for (unsigned int i = 0; i<6; i++)
	{
		for (unsigned int j = i; j<6; j++)
		{
			res(j, i) = res(i, j);
		}
	}

	conf.numCorr = (unsigned int)numCorrF;

	return res;
}
