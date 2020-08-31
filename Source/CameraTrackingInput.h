#include "cudaUtil.h"

#pragma once


struct CameraTrackingParameters
{
	 float weightDepth;
	 float weightColor;
	 float distThres;
	 float normalThres;

	 float sensorMaxDepth;
	 float colorGradiantMin;
	 float colorThres;
};

struct CameraTrackingInput
{
	float4* d_inputPos;
	float4* d_inputNormal;
	float*  d_inputIntensity;
	float4* d_targetPos;
	float4* d_targetNormal;
	float4* d_targetIntensityAndDerivatives;
};


struct CameraTrackingLocalParameters
{

	float lambdaReg;
	int localWindowHWidth;

	float sensorMaxDepth;
	float colorGradiantMin;
	float colorThres;

	float2 offset;
	float2 cellWH;
	float sigma;
	int nodeWidth, nodeHeight;
	int imageWidth, imageHeight;
};

struct CameraTrackingLocalInput
{
	float4* d_inputPos;
	float4* d_inputNormal;
	float*  d_inputIntensity;
	float* d_inputMask;
	float* d_targetMask;
	float4* d_targetIntensityAndDerivatives;
};