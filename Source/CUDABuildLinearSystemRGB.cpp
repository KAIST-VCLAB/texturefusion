#include "stdafx.h"

#include "CUDABuildLinearSystemRGB.h"

#include "GlobalAppState.h"
#include "GlobalCameraTrackingState.h"

#include "MatrixConversion.h"

#include <iostream>

#define DEBUG_TRACKING


extern "C" void computeNormalEquationsRGB(unsigned int imageWidth, unsigned int imageHeight, float* output, CameraTrackingInput cameraTrackingInput, float* intrinsics, CameraTrackingParameters cameraTrackingParameters, float3 anglesOld, float3 translationOld, unsigned int localWindowSize, unsigned int blockSize);


CUDABuildLinearSystemRGB::CUDABuildLinearSystemRGB(unsigned int imageWidth, unsigned int imageHeight) 
{
	cutilSafeCall(cudaMalloc(&d_output, 30*sizeof(float)*imageWidth*imageHeight));
	h_output = new float[30*imageWidth*imageHeight];
}

CUDABuildLinearSystemRGB::~CUDABuildLinearSystemRGB() {
	if (d_output) {
		cutilSafeCall(cudaFree(d_output));
	}
	if (h_output) {
		SAFE_DELETE_ARRAY(h_output);
	}
}

void CUDABuildLinearSystemRGB::applyBL(CameraTrackingInput cameraTrackingInput, Eigen::Matrix3f& intrinsics, CameraTrackingParameters cameraTrackingParameters, float3 anglesOld, float3 translationOld, unsigned int imageWidth, unsigned int imageHeight, unsigned int level, Matrix6x7f& res, LinearSystemConfidence& conf) 
{
	unsigned int localWindowSize = 12;
	if(level != 0) localWindowSize = max(1, localWindowSize/(4*level));

	const unsigned int blockSize = 64;
	const unsigned int dimX = (unsigned int)ceil(((float)imageWidth*imageHeight)/(localWindowSize*blockSize));

#ifdef DEBUG_TRACKING
	
	printf("localWindowSize: %d\n", localWindowSize);
	printf("blockSze: %d\n", blockSize);
	printf("image width image height: %d %d\n", imageWidth, imageHeight);

#endif

	Eigen::Matrix3f intrinsicsRowMajor = intrinsics.transpose(); // Eigen is col major / cuda is row major
	computeNormalEquationsRGB(imageWidth, imageHeight, d_output, cameraTrackingInput, intrinsicsRowMajor.data(), cameraTrackingParameters, anglesOld, translationOld, localWindowSize, blockSize);

	cutilSafeCall(cudaMemcpy(h_output, d_output, sizeof(float)*30*dimX, cudaMemcpyDeviceToHost));

	// Copy to CPU
	res = reductionSystemCPU(h_output, dimX, conf);

}

Matrix6x7f CUDABuildLinearSystemRGB::reductionSystemCPU( const float* data, unsigned int nElems, LinearSystemConfidence& conf )
{
	Matrix6x7f res; res.setZero();

	conf.reset();
	float numCorrF = 0.0f;

	for(unsigned int k = 0; k<nElems; k++)
	{
		unsigned int linRowStart = 0;

		for(unsigned int i = 0; i<6; i++)
		{
			for(unsigned int j = i; j<6; j++)
			{
				res(i, j) += data[30*k+linRowStart+j-i];
			}

			linRowStart += 6-i;

			res(i, 6) += data[30*k+21+i];
		}

		conf.sumRegError += data[30*k+27];
		conf.sumRegWeight += data[30*k+28];

		numCorrF += data[30*k+29];
	}

	// Fill lower triangle
	for(unsigned int i = 0; i<6; i++)
	{
		for(unsigned int j = i; j<6; j++)
		{
			res(j, i) = res(i, j);
		}
	}

	conf.numCorr = (unsigned int)numCorrF;

	return res;
}
