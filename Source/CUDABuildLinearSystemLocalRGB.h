#pragma once

#include "stdafx.h"

#include "Eigen.h"

#include <cutil_inline.h>
#include <cutil_math.h>

#include "CameraTrackingInput.h"
#include "ICPErrorLog.h"


class CUDABuildLinearSystemLocalRGB
{
public:

	CUDABuildLinearSystemLocalRGB(unsigned int imageWidth, unsigned int imageHeight);
	CUDABuildLinearSystemLocalRGB(unsigned int imageWidth, unsigned int imageHeight, unsigned int nodeWidth, unsigned int nodeHeight);
	~CUDABuildLinearSystemLocalRGB();

	void applyBL(CameraTrackingLocalInput cameraTrackingInput, Eigen::Matrix3f& intrinsics, CameraTrackingLocalParameters cameraTrackingParameters, float3 anglesOld, float3 translationOld, float2 poseCenter, unsigned int imageWidth, unsigned int imageHeight, unsigned int level, Matrix6x7f& res, LinearSystemConfidence& conf);

	void applyBLs(CameraTrackingLocalInput cameraTrackingInput, Eigen::Matrix3f& intrinsics, CameraTrackingLocalParameters cameraTrackingParameters, float3 *d_x_rot, float3 *d_x_trans, float3 *d_x_step_rot, float3 *d_x_step_trans, float lambdaReg, float *d_system, unsigned int level, LinearSystemConfidence& conf);



	//! builds AtA, AtB, and confidences
	Matrix6x7f reductionSystemCPU(const float* data, unsigned int nElems, LinearSystemConfidence& conf);

private:

	float* d_output;
	float *d_temp;
	float* h_output;
};
