#include "stdafx.h"

#include "modeDefine.h"
#include "CUDACameraTrackingLocalRGB.h"
#include "CUDAImageHelper.h"

#include "GlobalAppState.h"

#include <cutil_inline.h>
#include <cutil_math.h>

#include <iostream>
#include <limits>
#include "cudaDebug.h"

#include "CUDAImageUtil.h"

extern "C" void resampleFloat4Map(float4* d_colorMapResampledFloat4, unsigned int outputWidth, unsigned int outputHeight, float4* d_colorMapFloat4, unsigned int inputWidth, unsigned int inputHeight);
extern "C" void resampleFloatMap(float* d_colorMapResampledFloat, unsigned int outputWidth, unsigned int outputHeight, float* d_colorMapFloat, unsigned int inputWidth, unsigned int inputHeight);

extern "C" void computeNormals(float4* d_output, float4* d_input, unsigned int width, unsigned int height);
extern "C" void computeDerivativesCameraSpace(float4* d_positions, unsigned int imageWidth, unsigned int imageHeight, float4* d_positionsDU, float4* d_positionsDV);

extern "C" void convertColorToIntensityFloat(float* d_output, float4* d_input, unsigned int width, unsigned int height);
extern "C" void convertColorToIntensityFloat4(float4* d_output, float4* d_input, unsigned int width, unsigned int height);

extern "C" void computeGradientIntensityMagnitude(float4* d_inputDU, float4* d_inputDV, unsigned int imageWidth, unsigned int imageHeight, float4* d_ouput);

extern "C" void computeIntensityAndDerivatives(float* d_intensity, unsigned int imageWidth, unsigned int imageHeight, float4* d_intensityAndDerivatives);

extern "C" void computeIntensityAndDerivativesMask(float* d_intensity, unsigned int imageWidth, unsigned int imageHeight, float4* d_intensityAndDerivatives, float *d_mask);

extern "C" void gaussFilterFloat4Map(float4* d_output, float4* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height);
extern "C" void gaussFilterFloatMap(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height);

extern "C" void copyFloatMap(float* d_output, float* d_input, unsigned int width, unsigned int height);

extern "C" void renderCorrespondenceLocalCUDA(unsigned int imageWidth, unsigned int imageHeight, float *output, CameraTrackingLocalInput cameraTrackingInput, float* intrinsics, CameraTrackingLocalParameters cameraTrackingIParameters, float* d_transforms);

extern "C" void renderCorrespondenceLocal2CUDA(unsigned int imageWidth, unsigned int imageHeight, float *output, CameraTrackingLocalInput cameraTrackingInput, float* intrinsics, CameraTrackingLocalParameters cameraTrackingIParameters, float3* d_x_rot, float3 *d_x_trans);

extern "C" void setTransform(float3 *d_xold_rot, float3 *d_xold_trans, float3 initrot, float3 inittrans, CameraTrackingLocalParameters cameraTrackingParameters); 

extern "C" void upsampleGrid(float3 *d_xold_rot, float3 *d_xold_trans, float3 *d_x_rot, float3 *d_x_trans, float2 offsetCur, float2 offset, float2 cellWHCur, float2 cellWH, int nodeWCur, int nodeW, int nodeHCur, int nodeH);

extern "C" void updateTransforms(float3 *d_x_rot, float3 *d_x_trans, float3 *d_xStep_rot, float3 *d_xStep_trans, float3 *d_xOld_rot, float3 *d_xOld_trans, int nodeN);



Timer CUDACameraTrackingMultiResLocalRGB::m_timer;

void print(float3 x) {
	printf("%f %f %f", x.x, x.y, x.z);

}

CUDACameraTrackingMultiResLocalRGB::CUDACameraTrackingMultiResLocalRGB(unsigned int imageWidth, unsigned int imageHeight, unsigned int levels, const std::vector<float>& offsetx, const std::vector<float>& offsety,
	const std::vector<float>& cellWidth,
	const std::vector<float>& cellHeight,
	const std::vector<int>& localWindowHWidth) {

	m_levels = levels;

	d_input = new float4*[m_levels];
	d_inputNormal = new float4*[m_levels];

	d_inputIntensity = new float*[m_levels];
	d_inputIntensityFiltered = new float*[m_levels];
	d_inputMask = new float*[m_levels];

	d_modelIntensity = new float*[m_levels];
	d_modelIntensityFiltered = new float*[m_levels];
	d_modelIntensityAndDerivatives = new float4*[m_levels];
	d_modelMask = new float*[m_levels];


	m_imageWidth = new unsigned int[m_levels];
	m_imageHeight = new unsigned int[m_levels];
	m_nodeWidth = new unsigned int[m_levels];
	m_nodeHeight = new unsigned int[m_levels];
	m_offset = new float2[m_levels];
	m_cellWH = new float2[m_levels];
	m_localWindowHWidth = new unsigned int[m_levels];

//	m_sigma = new float[m_levels];

	cudaMalloc(&d_warpedModelIntensity, sizeof(float) * imageWidth * imageHeight);

	unsigned int fac = 1;
	for (unsigned int i = 0; i < m_levels; i++) {
		m_imageWidth[i] = imageWidth / fac;
		m_imageHeight[i] = imageHeight / fac;

		m_offset[i].x = offsetx[i];
		m_offset[i].y = offsety[i];
		m_cellWH[i].x = cellWidth[i];
		m_cellWH[i].y = cellHeight[i];
		m_localWindowHWidth[i] = localWindowHWidth[i];

		m_nodeWidth[i] = (int)((m_imageWidth[i] - m_offset[i].x + m_cellWH[i].x - 1) / m_cellWH[i].x);
		m_nodeHeight[i] = (int)((m_imageHeight[i] - m_offset[i].y + m_cellWH[i].y - 1) / m_cellWH[i].y);

		printf("m_imageWidth, m_imageHeight : %d %d\n\n", m_imageWidth[i], m_imageHeight[i]);
		printf("nodew, nodeh : %d %d\n\n", m_nodeWidth[i], m_nodeHeight[i]);
		printf("m_offset xy : %f %f\n\n", m_offset[i].x, m_offset[i].y);
		printf("m_cellWH xy : %f %f\n\n", m_cellWH[i].x, m_cellWH[i].y);

		// input
		if (i != 0) {  // Not finest level
			cutilSafeCall(cudaMalloc(&d_input[i], sizeof(float4)*m_imageWidth[i] * m_imageHeight[i]));
			cutilSafeCall(cudaMalloc(&d_inputNormal[i], sizeof(float4)*m_imageWidth[i] * m_imageHeight[i]));
		}
		else {
			d_input[i] = NULL;
			d_inputNormal[i] = NULL;
		}

		cutilSafeCall(cudaMalloc(&d_inputIntensity[i], sizeof(float)*m_imageWidth[i] * m_imageHeight[i]));
		cutilSafeCall(cudaMalloc(&d_inputIntensityFiltered[i], sizeof(float)*m_imageWidth[i] * m_imageHeight[i]));
		cutilSafeCall(cudaMalloc(&d_inputMask[i], sizeof(float)*m_imageWidth[i] * m_imageHeight[i]));

		// model
		cutilSafeCall(cudaMalloc(&d_modelMask[i], sizeof(float)*m_imageWidth[i] * m_imageHeight[i]));
		cutilSafeCall(cudaMalloc(&d_modelIntensity[i], sizeof(float)*m_imageWidth[i] * m_imageHeight[i]));
		cutilSafeCall(cudaMalloc(&d_modelIntensityFiltered[i], sizeof(float)*m_imageWidth[i] * m_imageHeight[i]));
		cutilSafeCall(cudaMalloc(&d_modelIntensityAndDerivatives[i], sizeof(float4)*m_imageWidth[i] * m_imageHeight[i]));

		fac *= 2;
	}

	cutilSafeCall(cudaMalloc(&d_transforms, m_nodeWidth[0] * m_nodeHeight[0] * 6 * sizeof(float)));
	h_transforms = (float*)malloc(sizeof(float) * 6 * m_nodeWidth[0] * m_nodeHeight[0]);

	cutilSafeCall(cudaMalloc(&d_x_map, m_imageWidth[0] * m_imageHeight[0] * 6 * sizeof(float)));
	cutilSafeCall(cudaMalloc(&d_x_rot, m_nodeWidth[0] * m_nodeHeight[0] * 3 * sizeof(float)));
	cutilSafeCall(cudaMalloc(&d_x_trans, m_nodeWidth[0] * m_nodeHeight[0] * 3 * sizeof(float)));
	cutilSafeCall(cudaMalloc(&d_xStep_rot, m_nodeWidth[0] * m_nodeHeight[0] * 3 * sizeof(float)));
	cutilSafeCall(cudaMalloc(&d_xStep_trans, m_nodeWidth[0] * m_nodeHeight[0] * 3 * sizeof(float)));
	cutilSafeCall(cudaMalloc(&d_xDelta_rot, m_nodeWidth[0] * m_nodeHeight[0] * 3 * sizeof(float)));
	cutilSafeCall(cudaMalloc(&d_xDelta_trans, m_nodeWidth[0] * m_nodeHeight[0] * 3 * sizeof(float)));
	cutilSafeCall(cudaMalloc(&d_xOld_rot, m_nodeWidth[0] * m_nodeHeight[0] * 3 * sizeof(float)));
	cutilSafeCall(cudaMalloc(&d_xOld_trans, m_nodeWidth[0] * m_nodeHeight[0] * 3 * sizeof(float)));

	m_CUDABuildLinearSystem = new CUDABuildLinearSystemLocalRGB(m_imageWidth[0], m_imageHeight[0], m_nodeWidth[0], m_nodeHeight[0]);

	h_system = (float*)malloc(sizeof(float) * 30 * m_nodeWidth[0] * m_nodeHeight[0]);
	h_xStep_rot = (float3*)malloc(sizeof(float3) * m_nodeWidth[0] * m_nodeHeight[0]);
	h_xStep_trans = (float3*)malloc(sizeof(float3) * m_nodeWidth[0] * m_nodeHeight[0]);
	h_xDelta_rot = (float3*)malloc(sizeof(float3) * m_nodeWidth[0] * m_nodeHeight[0]);
	h_xDelta_trans = (float3*)malloc(sizeof(float3) * m_nodeWidth[0] * m_nodeHeight[0]);
	cudaMalloc(&d_system, sizeof(float) * 30 * m_nodeWidth[0] * m_nodeHeight[0]);

}

bool CUDACameraTrackingMultiResLocalRGB::checkRigidTransformation(Eigen::Matrix3f& R, Eigen::Vector3f& t, float angleThres, float distThres) {
	Eigen::AngleAxisf aa(R);

	if (aa.angle() > angleThres || t.norm() > distThres) {
		std::cout << "Tracking lost: angle " << (aa.angle() / M_PI)*180.0f << " translation " << t.norm() << std::endl;
		return false;
	}

	return true;
}

Eigen::Matrix4f CUDACameraTrackingMultiResLocalRGB::delinearizeTransformation(Vector6f& x, Eigen::Vector3f& mean, float meanStDev, unsigned int level)
{
	Eigen::Matrix3f R = Eigen::AngleAxisf(x[0], Eigen::Vector3f::UnitZ()).toRotationMatrix()  // Rot Z
		*Eigen::AngleAxisf(x[1], Eigen::Vector3f::UnitY()).toRotationMatrix()  // Rot Y
		*Eigen::AngleAxisf(x[2], Eigen::Vector3f::UnitX()).toRotationMatrix(); // Rot X

	Eigen::Vector3f t = x.segment(3, 3);

	Eigen::Matrix4f res; res.setIdentity();

	//if (!checkRigidTransformation(R, t, GlobalCameraTrackingState::getInstance().s_angleTransThres[level], GlobalCameraTrackingState::getInstance().s_distTransThres[level])) {
	//	return res;
	//}

	res.block(0, 0, 3, 3) = R;
	res.block(0, 3, 3, 1) = meanStDev*t + mean - R*mean;

	return res;
}

CUDACameraTrackingMultiResLocalRGB::~CUDACameraTrackingMultiResLocalRGB(){

	d_input[0] = NULL;
	d_inputNormal[0] = NULL;

	if (h_system) free(h_system);
	if (h_xStep_rot) free(h_xStep_rot);
	if (h_xStep_trans) free(h_xStep_trans);
	if (d_system) cudaFree(d_system);

	// input
	if (d_input) {
		for (unsigned int i = 0; i < m_levels; i++)
			if (d_input[i])	cutilSafeCall(cudaFree(d_input[i]));
		SAFE_DELETE_ARRAY(d_input)
	}

	if (d_inputNormal) {
		for (unsigned int i = 0; i < m_levels; i++)
			if (d_inputNormal[i])	cutilSafeCall(cudaFree(d_inputNormal[i]));
		SAFE_DELETE_ARRAY(d_inputNormal)
	}

	if (d_inputMask) {
		for (unsigned int i = 0; i < m_levels; i++)
			if (d_inputMask[i])	cutilSafeCall(cudaFree(d_inputMask[i]));
		SAFE_DELETE_ARRAY(d_inputMask)
	}

	if (d_inputIntensity) {
		for (unsigned int i = 0; i < m_levels; i++)
			if (d_inputIntensity[i])	cutilSafeCall(cudaFree(d_inputIntensity[i]));
		SAFE_DELETE_ARRAY(d_inputIntensity)
	}

	if (d_inputIntensityFiltered) {
		for (unsigned int i = 0; i < m_levels; i++)
			if (d_inputIntensityFiltered[i])	cutilSafeCall(cudaFree(d_inputIntensityFiltered[i]));
		SAFE_DELETE_ARRAY(d_inputIntensityFiltered)
	}

	// model
	if (d_modelIntensity) {
		for (unsigned int i = 0; i < m_levels; i++)
			if (d_modelIntensity[i]) cutilSafeCall(cudaFree(d_modelIntensity[i]));
		SAFE_DELETE_ARRAY(d_modelIntensity)
	}

	if (d_modelIntensityFiltered) {
		for (unsigned int i = 0; i < m_levels; i++)
			if (d_modelIntensityFiltered[i]) cutilSafeCall(cudaFree(d_modelIntensityFiltered[i]));
		SAFE_DELETE_ARRAY(d_modelIntensityFiltered)
	}

	if (d_modelIntensityAndDerivatives) {
		for (unsigned int i = 0; i < m_levels; i++)
			if (d_modelIntensityAndDerivatives[i]) cutilSafeCall(cudaFree(d_modelIntensityAndDerivatives[i]));
		SAFE_DELETE_ARRAY(d_modelIntensityAndDerivatives)
	}

	if (m_imageWidth)	SAFE_DELETE_ARRAY(m_imageWidth);
	if (m_imageHeight)	SAFE_DELETE_ARRAY(m_imageHeight);

	if(d_transforms)
		cudaFree(d_transforms);
	SAFE_DELETE(m_CUDABuildLinearSystem);
	if (h_transforms)
		free(h_transforms);
}

void CUDACameraTrackingMultiResLocalRGB::applyMovingDLTOurs(
	float4* dInputPos, float4* dInputNormal, float4* dInputColor, float *dInputMask,
	float4* dTargetColor, float *dTargetMask,
	const mat4f& lastTransform, const std::vector<unsigned int>& maxInnerIter, const std::vector<unsigned int>& maxOuterIter,
	const std::vector<float>& colorGradiantMin,
	const std::vector<float>& colorThres,
	float condThres, float angleThres,
	const mat4f& deltaTransformEstimate,
	const std::vector<float>& lambdaReg,
	const std::vector<float>& sigma,
	const std::vector<float>& earlyOutResidual,
	const mat4f& intrinsic, const DepthCameraData& depthCameraData
) {
	// Input
	d_input[0] = dInputPos;
	d_inputNormal[0] = dInputNormal;
	d_inputMask[0] = dInputMask;
	d_modelMask[0] = dTargetMask;

	//Start Timing
	if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

	m_timer.start();

	//Color to gray
	convertColorToIntensityFloat(d_inputIntensity[0], dInputColor, m_imageWidth[0], m_imageHeight[0]);
	convertColorToIntensityFloat(d_modelIntensity[0], dTargetColor, m_imageWidth[0], m_imageHeight[0]);

	//compute gradient
	computeIntensityAndDerivativesMask(d_modelIntensity[0], m_imageWidth[0], m_imageHeight[0], 
		d_modelIntensityAndDerivatives[0], d_modelMask[0]);
	
	//copy image
	copyFloatMap(d_inputIntensityFiltered[0], d_inputIntensity[0], m_imageWidth[0], m_imageHeight[0]);
	
	for (unsigned int i = 0; i < m_levels - 1; i++)
	{
		float sigmaD = 3.0f; float sigmaR = 1.0f;

		//compute next-level images

		//downsample positions
		resampleFloat4Map(d_input[i + 1], m_imageWidth[i + 1], m_imageHeight[i + 1], d_input[i], m_imageWidth[i], m_imageHeight[i]);
		//compute normals from postions
		computeNormals(d_inputNormal[i + 1], d_input[i + 1], m_imageWidth[i + 1], m_imageHeight[i + 1]);
		
		// downsample a texture image
		resampleFloatMap(d_inputIntensity[i + 1], m_imageWidth[i + 1], m_imageHeight[i + 1], d_inputIntensity[i], m_imageWidth[i], m_imageHeight[i]);
		// blur the texture image
		gaussFilterFloatMap(d_inputIntensityFiltered[i + 1], d_inputIntensity[i + 1], sigmaD, sigmaR, m_imageWidth[i + 1], m_imageHeight[i + 1]);

		// downsample a captured image
		resampleFloatMap(d_modelIntensity[i + 1], m_imageWidth[i + 1], m_imageHeight[i + 1], d_modelIntensity[i], m_imageWidth[i], m_imageHeight[i]);
		// blur the captured image
		gaussFilterFloatMap(d_modelIntensityFiltered[i + 1], d_modelIntensity[i + 1], sigmaD, sigmaR, m_imageWidth[i + 1], m_imageHeight[i + 1]);

		// downsample masks
		downSampleMask(d_modelMask[i + 1], d_modelMask[i], m_imageWidth[i + 1], m_imageHeight[i + 1]);
		downSampleMask(d_inputMask[i + 1], d_inputMask[i], m_imageWidth[i + 1], m_imageHeight[i + 1]);
		//computeIntensityAndDerivatives(d_modelIntensityFiltered[i + 1], m_imageWidth[i + 1], m_imageHeight[i + 1], d_modelIntensityAndDerivatives[i + 1]);
		//	computeIntensityAndDerivativesMask(d_modelIntensityFiltered[i + 1], m_imageWidth[i + 1], m_imageHeight[i + 1], d_modelIntensityAndDerivatives[i + 1], d_modelMask[i+1]);
		
		//compute gradients of a captured image
		computeIntensityAndDerivativesMask(d_modelIntensity[i + 1], m_imageWidth[i + 1], m_imageHeight[i + 1], d_modelIntensityAndDerivatives[i + 1], d_modelMask[i + 1]);

	}

	//intialize parameters
	cudaMemset(d_x_rot, 0, sizeof(float3)*m_nodeHeight[0] * m_nodeWidth[0]);
	cudaMemset(d_x_trans, 0, sizeof(float3)*m_nodeHeight[0] * m_nodeWidth[0]);
	cudaMemset(d_xOld_rot, 0, sizeof(float3)*m_nodeHeight[0] * m_nodeWidth[0]);
	cudaMemset(d_xOld_trans, 0, sizeof(float3)*m_nodeHeight[0] * m_nodeWidth[0]);


	Eigen::Matrix4f deltaTransform; deltaTransform = MatToEig(deltaTransformEstimate);
	Eigen::Matrix3f R = deltaTransform.block(0, 0, 3, 3);
	Eigen::Vector3f eulerAngles = R.eulerAngles(2, 1, 0);

	float3 initRot, initTrans;

	initRot.x = eulerAngles(0);
	initRot.y = eulerAngles(1);
	initRot.z = eulerAngles(2);

	initTrans.x = deltaTransform(0,3);
	initTrans.y = deltaTransform(1,3);
	initTrans.z = deltaTransform(2,3);

	for (int level = m_levels - 1; level >= 0; level--)
	{
//		printf("level: %d\n", level);

		float levelFactor = pow(2.0f, (float)level);
		mat4f intrinsicNew = intrinsic;
		intrinsicNew(0, 0) /= levelFactor; intrinsicNew(1, 1) /= levelFactor; intrinsicNew(0, 2) /= levelFactor; intrinsicNew(1, 2) /= levelFactor;

		CameraTrackingLocalInput input;
		input.d_inputPos = d_input[level];
		input.d_inputNormal = d_inputNormal[level];
//		input.d_inputIntensity = d_inputIntensityFiltered[level];
		input.d_inputIntensity = d_inputIntensity[level];
		input.d_targetMask = d_modelMask[level];
		input.d_inputMask = d_inputMask[level];
		input.d_targetIntensityAndDerivatives = d_modelIntensityAndDerivatives[level];

		CameraTrackingLocalParameters parameters;
		parameters.lambdaReg = lambdaReg[level];
		parameters.localWindowHWidth = m_localWindowHWidth[level];
		parameters.sensorMaxDepth = GlobalAppState::get().s_sensorDepthMax;
		parameters.colorGradiantMin = colorGradiantMin[level];
		parameters.colorThres = colorThres[level];

		parameters.offset = m_offset[level];
		parameters.cellWH = m_cellWH[level];
		parameters.sigma = sigma[level];
		parameters.nodeWidth = m_nodeWidth[level];
		parameters.nodeHeight = m_nodeHeight[level];
		parameters.imageWidth = m_imageWidth[level];
		parameters.imageHeight = m_imageHeight[level];

		if (level == m_levels - 1) {

			setTransform(d_xOld_rot, d_xOld_trans, initRot, initTrans, parameters);
		
		}
		else {

			upsampleGrid(d_xOld_rot, d_xOld_trans, d_x_rot, d_x_trans, m_offset[level], m_offset[level + 1], m_cellWH[level], m_cellWH[level + 1], m_nodeWidth[level], m_nodeWidth[level + 1], m_nodeHeight[level], m_nodeHeight[level + 1]);

		}

		alignParallel(input, level, parameters, maxInnerIter[level], maxOuterIter[level], condThres, angleThres, earlyOutResidual[level], intrinsicNew, depthCameraData);

#ifdef SHOW_LOCALLYVARYING_CORRECTION
		renderCorrespondenceOurs(d_x_rot, d_x_trans, input, level, parameters, intrinsicNew, depthCameraData);

		displayFloatMat(d_warpedModelIntensity, "PCG", m_imageWidth[level], m_imageHeight[level]);

		/*displayFloatMat(d_modelIntensity[level], " model intensity (local)", m_imageWidth[level], m_imageHeight[level]);*/


			displayDiffFloatMat(d_warpedModelIntensity, input.d_inputIntensity, "ourDiff", m_imageWidth[level], m_imageHeight[level], 0.5, 0.5);

			displayDiffFloatMat(d_modelIntensity[level], d_inputIntensity[level], "groundDiff", m_imageWidth[level], m_imageHeight[level], 0.5, 0.5);

			cv::waitKey(0);
#endif
		
	}

	m_timer.stop();

}

void CUDACameraTrackingMultiResLocalRGB::upsample(std::vector<Vector6f> &deltaTransformsCur, std::vector<Vector6f> &deltaTransforms, float2 offsetCur, float2 offset, float2 cellWHCur, float2 cellWH, int nodeWCur, int nodeW, int nodeHCur, int nodeH) {

	int nodeNCur = nodeWCur * nodeHCur;
	int nodeN = nodeW * nodeH;


	for (int i = 0; i < nodeN; i++) {
		Vector6f res;
		res = deltaTransforms[i];
	}

	for (int i = 0; i < nodeNCur; i++) {

		int nodex = i % nodeWCur;
		int nodey = i / nodeWCur;

		//compute the point on prev space
		float2 posCur = offsetCur + make_float2(cellWHCur.x * nodex, cellWHCur.y*nodey);

		posCur *= 0.5;// multiplied by factor
		float2 posCurLocal = (posCur - offset)/cellWH;


		if (posCurLocal.x > nodeW - 1) posCurLocal.x = nodeW - 1.000001;
		if (posCurLocal.x < 0) posCurLocal.x = 0.000001;
		if (posCurLocal.y > nodeH - 1) posCurLocal.y = nodeH - 1.000001;
		if (posCurLocal.y < 0) posCurLocal.y = 0.000001;

		int2 posi = make_int2(posCurLocal);
		int index = posi.y * nodeW + posi.x;

		float2 weight = make_float2 (posCurLocal.x - (int)posCurLocal.x, posCurLocal.y - (int)posCurLocal.y);

		Vector6f ld, rd, lu, ru;
		ld = deltaTransforms[index];
		rd = Vector6f::Zero();
		lu = Vector6f::Zero();
		ru = Vector6f::Zero();

		if (posi.x < nodeW-1)
			rd = deltaTransforms[index + 1];
		if (posi.y < nodeH-1)
			lu = deltaTransforms[index + nodeW ];
		if (posi.x < nodeW-1&& posi.y < nodeH-1)
			ru = deltaTransforms[index + nodeW +1];


		Vector6f res = (1 - weight.x) * (1 - weight.y) *ld +
			weight.x * (1-weight.y) * rd +
			(1-weight.x) * weight.y * lu +
			weight.x * weight.y * ru;

		deltaTransformsCur[i] = res;

	}
}

//used
void CUDACameraTrackingMultiResLocalRGB::alignParallel(CameraTrackingLocalInput cameraTrackingInput, unsigned int level, CameraTrackingLocalParameters cameraTrackingParameters, unsigned int maxInnerIter, unsigned maxOuterIter, float condThres, float angleThres, float earlyOut, const mat4f& intrinsic, const DepthCameraData& depthCameraData)
{
	float lastICPError = -1.0f;
	const int nodeN = cameraTrackingParameters.nodeWidth* cameraTrackingParameters.nodeHeight;

	cudaMemset(d_xStep_rot, 0, sizeof(float3)*nodeN);
	cudaMemset(d_xStep_trans, 0, sizeof(float3)*nodeN);


	for (unsigned int i = 0; i < maxOuterIter; i++)
	{
		LinearSystemConfidence currConf;

		Eigen::Matrix4f intrinsics4x4 = MatrixConversion::MatToEig(intrinsic);
		Eigen::Matrix3f intrinsics = intrinsics4x4.block(0, 0, 3, 3);


		computeBestRigidAlignmentParallel(cameraTrackingInput, intrinsics, level, cameraTrackingParameters, maxInnerIter, condThres, angleThres, currConf);
		
	}

	updateTransforms(d_x_rot, d_x_trans, d_xOld_rot, d_xOld_trans, d_xStep_rot, d_xStep_trans, nodeN);


}

Matrix6x7f CUDACameraTrackingMultiResLocalRGB::reductionSystemCPU(int k , const float* data, LinearSystemConfidence& conf)
{
	Matrix6x7f res; res.setZero();

	conf.reset();
	float numCorrF = 0.0f;
		unsigned int linRowStart = 0;

		for (unsigned int i = 0; i<6; i++)
		{
			for (unsigned int j = i; j<6; j++)
			{
				res(i, j) = data[30 * k + linRowStart + j - i];
			}

			linRowStart += 6 - i;

			res(i, 6) = data[30 * k + 21 + i];
		}

		conf.sumRegError = data[30 * k + 27];
		conf.sumRegWeight = data[30 * k + 28];

		numCorrF = data[30 * k + 29];

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

//used
void CUDACameraTrackingMultiResLocalRGB::computeBestRigidAlignmentParallel(CameraTrackingLocalInput cameraTrackingInput, Eigen::Matrix3f& intrinsics, unsigned int level, CameraTrackingLocalParameters cameraTrackingParameters, unsigned int maxInnerIter, float condThres, float angleThres, LinearSystemConfidence& conf) {


	//Eigen::Matrix4f deltaTransform = globalDeltaTransform;
	conf.reset();

	int nodeN = cameraTrackingParameters.nodeWidth * cameraTrackingParameters.nodeHeight;

	//m_timer.start();

	//compute matrix
	m_CUDABuildLinearSystem->applyBLs(cameraTrackingInput, intrinsics, cameraTrackingParameters, d_xOld_rot, d_xOld_trans, d_xStep_rot, d_xStep_trans, cameraTrackingParameters.lambdaReg,  d_system, level, conf);

	// compute delta ( SVD )
	cudaMemcpy(h_system, d_system, sizeof(float) * 30 * nodeN, cudaMemcpyDeviceToHost);

	Matrix6x7f system;

	//m_timer.start();

	for (int i = 0; i < nodeN; i++) {

		system = reductionSystemCPU(i, h_system, conf);

		Matrix6x6f ATA = system.block(0, 0, 6, 6);
		Vector6f ATb = system.block(0, 6, 6, 1);

		if (ATA.isZero()) {

		}
		else {

			Eigen::JacobiSVD<Matrix6x6f> SVD(ATA, Eigen::ComputeFullU | Eigen::ComputeFullV);
			Vector6f x = SVD.solve(ATb);

			//computing the matrix condition
			Vector6f evs = SVD.singularValues();
			conf.matrixCondition = evs[0] / evs[5];


			Eigen::Matrix4f deltaTransform = delinearizeTransformation(x, Eigen::Vector3f(0.0f, 0.0f, 0.0f), 1.0f, level);
			
			if (deltaTransform(0, 0) != -std::numeric_limits<float>::infinity())
			{
				h_xDelta_rot[i].x = x(0);
				h_xDelta_rot[i].y = x(1);
				h_xDelta_rot[i].z = x(2);
				h_xDelta_trans[i].x = x(3);
				h_xDelta_trans[i].y = x(4);
				h_xDelta_trans[i].z = x(5);
			}
			else {
				printf("inifinity\n");
				h_xDelta_rot[i].x =0.f;
				h_xDelta_rot[i].y = 0.f;
				h_xDelta_rot[i].z = 0.f;
				h_xDelta_trans[i].x = 0.f;
				h_xDelta_trans[i].y = 0.f;
				h_xDelta_trans[i].z = 0.f;
			}
		}
	}

	// update delta 
	cudaMemcpy(d_xDelta_rot, h_xDelta_rot, sizeof(float) * 3 * nodeN, cudaMemcpyHostToDevice);
	cudaMemcpy(d_xDelta_trans, h_xDelta_trans, sizeof(float) * 3 * nodeN, cudaMemcpyHostToDevice);
	
	updateTransforms(d_xStep_rot, d_xStep_trans, d_xStep_rot, d_xStep_trans, d_xDelta_rot, d_xDelta_trans, cameraTrackingParameters.nodeHeight *cameraTrackingParameters.nodeWidth);
	
}

void CUDACameraTrackingMultiResLocalRGB::renderCorrespondenceOurs(float3 *d_x_rots, float3 *d_x_transs, CameraTrackingLocalInput cameraTrackingInput, unsigned int level, CameraTrackingLocalParameters cameraTrackingParameters, const mat4f& intrinsic, const DepthCameraData& depthCameraData) {

	Eigen::Matrix4f intrinsics4x4 = MatrixConversion::MatToEig(intrinsic);
	Eigen::Matrix3f intrinsics = intrinsics4x4.block(0, 0, 3, 3);
	Eigen::Matrix3f intrinsicsRowMajor = intrinsics.transpose();


	int nodeN = cameraTrackingParameters.nodeWidth *cameraTrackingParameters.nodeHeight;

	renderCorrespondenceLocal2CUDA(m_imageWidth[level], m_imageHeight[level], d_warpedModelIntensity, cameraTrackingInput, intrinsicsRowMajor.data(), cameraTrackingParameters, d_x_rots, d_x_transs);

}

extern"C" void renderMotionMapCUDA(float * d_x_map, float3 *d_x_rot, float3 *d_x_trans, float2 offset, float2 cellWH, int imageWidth, int imageHeight, int nodeWidth, int nodeHeight);

void CUDACameraTrackingMultiResLocalRGB::renderMotionMap() {

	renderMotionMapCUDA(d_x_map, d_x_rot, d_x_trans, m_offset[0], m_cellWH[0], m_imageWidth[0], m_imageHeight[0], m_nodeWidth[0], m_nodeHeight[0]);

}

float *CUDACameraTrackingMultiResLocalRGB::getMotionMap() {
	return d_x_map;
}

void CUDACameraTrackingMultiResLocalRGB::writeFinalResult(char *filepfx, const mat4f& intrinsic, const DepthCameraData& depthCameraData) {

	int level = 0;
	char filename[30];

	float levelFactor = pow(2.0f, (float)level);
	mat4f intrinsicNew = intrinsic;
	intrinsicNew(0, 0) /= levelFactor; intrinsicNew(1, 1) /= levelFactor; intrinsicNew(0, 2) /= levelFactor; intrinsicNew(1, 2) /= levelFactor;

	CameraTrackingLocalInput input;
	input.d_inputPos = d_input[level];
	input.d_inputNormal = d_inputNormal[level];
	//		input.d_inputIntensity = d_inputIntensityFiltered[level];
	input.d_inputIntensity = d_inputIntensity[level];
	input.d_targetMask = d_modelMask[level];
	input.d_inputMask = d_inputMask[level];
	input.d_targetIntensityAndDerivatives = d_modelIntensityAndDerivatives[level];

	CameraTrackingLocalParameters parameters;
	parameters.localWindowHWidth = m_localWindowHWidth[level];
	parameters.sensorMaxDepth = GlobalAppState::get().s_sensorDepthMax;

	parameters.offset = m_offset[level];
	parameters.cellWH = m_cellWH[level];
	parameters.nodeWidth = m_nodeWidth[level];
	parameters.nodeHeight = m_nodeHeight[level];
	parameters.imageWidth = m_imageWidth[level];
	parameters.imageHeight = m_imageHeight[level];

	renderCorrespondenceOurs(d_x_rot, d_x_trans, input, level, parameters, intrinsicNew, depthCameraData);

	sprintf(filename, "%s_op_render.png", filepfx);
	writeFloatMat((float*)d_warpedModelIntensity, std::string(filename), parameters.imageWidth, parameters.imageHeight, 1);
	sprintf(filename, "%s_op_model.png", filepfx);
	writeFloatMat((float*)input.d_inputIntensity, std::string(filename), parameters.imageWidth, parameters.imageHeight, 1);
	sprintf(filename, "%s_op_capture.png", filepfx);
	writeFloatMat((float*)d_modelIntensity[0], std::string(filename), parameters.imageWidth, parameters.imageHeight, 1);


	cv::waitKey(0);

}