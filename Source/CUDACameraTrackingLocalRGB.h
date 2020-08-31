     
#include "DepthCameraUtil.h"
#include "MatrixConversion.h"
#include "DX11QuadDrawer.h"
#include "CUDABuildLinearSystemLocalRGB.h"
#include "ICPErrorLog.h"
#include "TimingLog.h"
#include "Eigen.h"

#include "CameraTrackingInput.h"

//#define SHOW_LOCALLYVARYING_CORRECTION

using namespace MatrixConversion;

class CUDACameraTrackingMultiResLocalRGB {

public:

	CUDACameraTrackingMultiResLocalRGB(unsigned int imageWidth, unsigned int imageHeight, unsigned int levels, const std::vector<float>& offsetx, const std::vector<float>& offsety,
		const std::vector<float>& cellWidth, const std::vector<float>& cellHeight,
		const std::vector<int>& localWindowHWidth );
	~CUDACameraTrackingMultiResLocalRGB();

	void applyMovingDLTOurs(
		float4* dInputPos, float4* dInputNormal, float4* dInputColor,  float *dInputMask,
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
	);

	void renderMotionMap();
	float *getMotionMap();
	void writeFinalResult(char *filename, const mat4f& intrinsic, const DepthCameraData& depthCameraData);

private:

	void renderCorrespondenceOurs(float3 *d_x_rots, float3 *d_x_transs, CameraTrackingLocalInput cameraTrackingInput, unsigned int level, CameraTrackingLocalParameters cameraTrackingParameters, const mat4f& intrinsic, const DepthCameraData& depthCameraData);
	
	void alignParallel(CameraTrackingLocalInput cameraTrackingInput, unsigned int level, CameraTrackingLocalParameters cameraTrackingParameters, unsigned int maxInnerIter, unsigned maxOuterIter, float condThres, float angleThres, float earlyOut, const mat4f& intrinsic, const DepthCameraData& depthCameraData);

	void computeBestRigidAlignmentParallel(CameraTrackingLocalInput cameraTrackingInput, Eigen::Matrix3f& intrinsics, unsigned int level, CameraTrackingLocalParameters cameraTrackingParameters, unsigned int maxInnerIter, float condThres, float angleThres, LinearSystemConfidence& conf);

	void upsample(std::vector<Vector6f> &deltaTransformsPrev, std::vector<Vector6f> &deltaTransforms, float2 offsetPrev, float2 offset, float2 cellWHPrev, float2 cellWH, int nodeWPrev, int nodeW, int nodeHPrev, int nodeH );

	Matrix6x7f reductionSystemCPU(int k, const float* data, LinearSystemConfidence& conf);

	bool checkRigidTransformation(Eigen::Matrix3f& R, Eigen::Vector3f& t, float angleThres, float distThres);

	Eigen::Matrix4f delinearizeTransformation(Vector6f& x, Eigen::Vector3f& mean, float meanStDev, unsigned int level);

	float4** d_input;
	float4** d_inputNormal;
	float**  d_inputIntensity;
	float**  d_inputIntensityFiltered;
	float**	 d_inputMask;

	//float4** d_model;
	//float4** d_modelNormal;
	float**  d_modelIntensity;
	float**  d_modelIntensityFiltered;
	float4** d_modelIntensityAndDerivatives;
	float*	 d_warpedModelIntensity;
	float**	 d_modelMask;

	// Image Pyramid Dimensions
	unsigned int* m_imageWidth;
	unsigned int* m_imageHeight;
	unsigned int* m_nodeWidth;
	unsigned int* m_nodeHeight;
	unsigned int* m_localWindowHWidth;

	float2 *m_offset;
	float2 *m_cellWH;
	//float* m_sigma;

	//float4x4 *d_trackingInput;
	float* d_transforms;
	float* h_transforms;

	float3* d_x_rot;
	float3* d_x_trans;
	float3* d_xStep_rot;//step
	float3* d_xStep_trans;
	float3* d_xDelta_rot;//step
	float3* d_xDelta_trans;
	float3* d_xOld_rot;
	float3* d_xOld_trans;
	float3* h_xStep_rot;
	float3* h_xStep_trans;
	float3* h_xDelta_rot;
	float3* h_xDelta_trans;
	float* h_system;
	float* d_system;

	float *d_x_map;
	float3* d_x_rotmap;
	float3* d_x_transmap;

	float m_lambda;

	unsigned int m_levels;
	
	CUDABuildLinearSystemLocalRGB* m_CUDABuildLinearSystem;

	static Timer m_timer;

};