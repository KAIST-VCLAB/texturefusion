#pragma once

#include "GlobalAppState.h"
#include "TimingLog.h"

#include "MatrixConversion.h"
#include "cuda_SimpleMatrixUtil.h"
#include "DepthCameraUtil.h"
#include "TexUpdateUtil.h"
#include "texturePool.h"
#include "VoxelUtilHashSDF.h"
#include "RayCastSDFUtil.h"
#include "CUDATexUpdateParams.h"

#include "DX11RayIntervalSplatting.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\core\cuda.hpp>
#include <opencv2\cudaoptflow.hpp>
#include <opencv2\cudaarithm.hpp>

#include "cudaDebug.h"

//#define SHOWMASK

extern"C" void resetColorCUDA(HashData& hashData, const HashParams& hashParams, const TexPoolData& texPoolData, const TexPoolParams& texPoolParams);
extern "C" void resetTexCUDA(TexPoolData& hashData, const TexPoolParams& hashParams);

extern "C" void computeOcclusionMaskCUDA (const RayCastData& rayCastData, const DepthCameraData& depthCameraData, const RayCastParams& rayCastParams, float *d_mask, int screenBoundaryWidth);
extern "C" void maskSlantRegionCUDA(const RayCastData& rayCastData, const DepthCameraData& depthCameraData, const RayCastParams& rayCastParams, float *d_mask);
extern "C" void computeSourceMaskCUDA(const RayCastData& rayCastData, const DepthCameraData& depthCameraData, const RayCastParams& rayCastParams, float *d_mask);
extern "C" void computeTargetMaskCUDA (const RayCastData& rayCastData, const DepthCameraData& depthCameraData, const RayCastParams& rayCastParams,float *d_mask);
extern "C" void computeBlendingMaskCUDA (const RayCastData& rayCastData, const DepthCameraData& depthCameraData, const RayCastParams& rayCastParams, float *d_mask);

extern "C" unsigned int findZeroCrossingVoxelsCUDA(HashData& hashData, const HashParams& hashParams, TexPoolData& texPoolData, const DepthCameraData& depthCameraData);
extern"C" void computeTexDepthCUDA(HashData& hashData, const HashParams& hashParams, TexPoolData& texPoolData, const TexPoolParams& texPoolParams, RayCastData& rayCastData, const DepthCameraData& cameraData);
extern "C" void deletePreviousTexTriangleCUDA(HashData& hashData, HashParams& hashParams, TexPoolData& texPoolData, const DepthCameraData& cameraData);

extern"C" void texUpdateFromPreviousTextureCUDA(HashData& hashData, const HashParams& hashParams, TexPoolData& texPoolData, const TexPoolParams& texPoolParams, RayCastData& rayCastData, const DepthCameraData& cameraData);

extern"C" void texUpdateFromImageCUDA(HashData& hashData, const HashParams& hasParams, TexUpdateData &texUpdateData, TexPoolData& texPoolData, const TexPoolParams& texPoolParams, RayCastData& rayCastData, const DepthCameraData& cameraData, float *d_blendingmask);

extern"C" void texUpdateFromImageHalfCUDA(HashData& hashData, const HashParams& hasParams, TexUpdateData &texUpdateData, TexPoolData& texPoolData, const TexPoolParams& texPoolParams, RayCastData& rayCastData, const DepthCameraData& cameraData, float *d_blendingmask);

extern"C" void texUpdateFromImageWithCameraMotionMapCUDA(HashData& hashData, const HashParams& hasParams, TexUpdateData &texUpdateData, TexPoolData& texPoolData, const TexPoolParams& texPoolParams, RayCastData& rayCastData, const DepthCameraData& cameraData, float *d_motion_map, float *d_occlusionweight);

extern"C" void texUpdateFromImageWithCameraMotionMapHalfCUDA(HashData& hashData, const HashParams& hasParams, TexUpdateData &texUpdateData, TexPoolData& texPoolData, const TexPoolParams& texPoolParams, RayCastData& rayCastData, const DepthCameraData& cameraData, float *d_motion_map, float *d_occlusionweight);

extern"C" void texUpdateFromImageOnlyCUDA(HashData& hashData, const HashParams& hasParams, TexUpdateData &texUpdateData, TexPoolData& texPoolData, const TexPoolParams& texPoolParams, RayCastData& rayCastData, const DepthCameraData& cameraData, float *d_mask);

extern"C" void updateMasksWithCapturedFrameCUDA(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, const float4 *d_normalmap, TexUpdateData& m_data);

void setMask(bool *d_mask, int image_w, int image_h, int offset);
void setMask(float *d_mask, int image_w, int image_h, int value, int offset);
void setMaskHost(bool *h_mask, int image_w, int image_h, int offset);
void setMaskHost(float *h_mask, int image_w, int image_h, int offset);

class CUDATexUpdate {
public:

	CUDATexUpdate(const TexUpdateParams& params) {
		create(params);
	}

	CUDATexUpdate(const TexUpdateParams& params, const TexPoolParams& texParams, const RayCastParams& rayCastParams) {

		//create(params);
		create(params, texParams, rayCastParams);

	}

	~CUDATexUpdate(void) {
		destroy();
	}
	
	float* getColorMask() {
		//return m_data.d_colorMask;
	}
	float* getSourceMask() {
		return m_data.d_sourceMask;
	}
	float* getTargetMask() {
		return m_data.d_targetMask;
	}
	float* getBlendingMask() {
		return m_data.d_blendingMask;
	}

	static TexUpdateParams parametersFromGlobalAppState(const GlobalAppState& gas) {
		TexUpdateParams params;

		params.m_warp_mode = gas.s_warpingMode;

		//erode
		params.m_erode_iter_stretch_box = gas.s_erodeIterStretchBox;
		params.m_erode_iter_stretch_gauss = gas.s_erodeIterStretchGauss;
		params.m_erode_sigma_stretch = gas.s_erodeSigmaStretch;
		params.m_screen_boundary_width = gas.s_screenBoundaryWidth;

		params.m_erode_iter_occdepth = gas.s_erodeIterOccDepth;

		params.m_width = gas.s_adapterWidth;
		params.m_height = gas.s_adapterHeight;

		params.m_angleThreshold_depth = gas.s_texAngleThreshold_depth;
		params.m_angleThreshold_update = gas.s_texAngleThreshold_update;
		params.m_integrationWeightMax = gas.s_texIntegrationWeightMax;
		params.m_integrationWeightSample = gas.s_texIntegrationWeightSample;

		params.m_sigma_angle = gas.s_sigmaAngle;
		params.m_sigma_depth = gas.s_sigmaDepth;
		params.m_sigma_area = gas.s_sigmaArea;

		return params;

	}

	static TexPoolParams texPoolParametersFromGlobalAppState(const GlobalAppState& gas) {

		TexPoolParams params;
		params.m_texturePatchWidth = gas.s_texPoolPatchWidth;
		params.m_texturePatchSize = params.m_texturePatchWidth * params.m_texturePatchWidth;
		params.m_numTexturePatches = gas.s_texPoolNumPatches;
		params.m_numTextureTileWidth = gas.s_numTextureTileWidth;

		params.m_minDepth = 0;
		params.m_maxDepth = gas.s_SDFVoxelSize;

		std::cout << "Patch size, patch width : " << params.m_texturePatchSize << " " << params.m_texturePatchWidth << std::endl;
		std::cout << "#Patches : " << params.m_numTexturePatches << std::endl;
		std::cout << "#TextureTils: " << params.m_numTextureTileWidth << std::endl;

		return params;
	}

	static RayCastParams rayCastParametersFromGlobalAppState(const GlobalAppState& gas) {
		RayCastParams params;
		
		params.m_minDepth = gas.s_sensorDepthMin; //
		params.m_maxDepth = gas.s_sensorDepthMax; //
		params.m_rayIncrement = gas.s_SDFRayIncrementFactor * gas.s_SDFTruncation; //
		params.m_thresSampleDist = gas.s_SDFRayThresSampleDistFactor * params.m_rayIncrement; //
		params.m_thresDist = gas.s_SDFRayThresDistFactor * params.m_rayIncrement; //
		params.m_useGradients = gas.s_SDFUseGradients; //

		return params;

	}

	void computeAlpha(int iter_n, float sigma);

	void erodeMask(float *d_mask, int erode_iter);

	void erodeMaskSoft(float *d_mask, int erode_iter, float sigma);
	
	void erodeDist(float *d_mask, int erode_iter);

	void extractTexture();
		
	//will be deprecated
	void computeOcclusionMask(const RayCastData &rayCastData, const DepthCameraData &depthCameraData, const RayCastParams &rayCastParams) {

		computeOcclusionMaskCUDA(rayCastData, depthCameraData, rayCastParams, m_data.d_occlusionMask, m_params.m_screen_boundary_width );



#ifdef SHOWMASK
		displayFloatMat(m_data.d_occlusionMask, "occlusionMask", rayCastParams.m_width, rayCastParams.m_height);
#endif
		

	}

	void computeSourceMask(const RayCastData &rayCastData, const DepthCameraData &depthCameraData, const RayCastParams &rayCastParams){

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_data.d_sourceMask, m_data.d_occlusionMask, sizeof(float) *rayCastParams.m_width*rayCastParams.m_height, cudaMemcpyDeviceToDevice));

		computeSourceMaskCUDA(rayCastData, depthCameraData, rayCastParams, m_data.d_sourceMask);
#ifdef SHOWMASK
		displayFloatMat(m_data.d_sourceMask, "sourceMask", rayCastParams.m_width, rayCastParams.m_height);
#endif

	}
	//used
	void computeTargetMask(const RayCastData &rayCastData, const DepthCameraData &depthCameraData, const RayCastParams &rayCastParams){
	
		cudaMemset( m_data.d_targetMask, 0, sizeof(float) * rayCastParams.m_width * rayCastParams.m_height );
	
#ifdef SHOWMASK
		displayFloatMat(m_data.d_targetMask, "target Mask", rayCastParams.m_width, rayCastParams.m_height);
#endif
	}


	void computeWarpingMask(const RayCastData &rayCastData, const DepthCameraData &depthCameraData, const RayCastParams &rayCastParams) {
		
		setMask(m_data.d_warpingMask, rayCastParams.m_width, rayCastParams.m_height, 1.f, 0);

#ifdef SHOWMASK
		displayFloatMat(m_data.d_warpingMask, "warping Mask", rayCastParams.m_width, rayCastParams.m_height);
#endif
	}

	void computeBlendingMask(const RayCastData &rayCastData, const DepthCameraData &depthCameraData, const RayCastParams &rayCastParams){

		cudaMemcpy(m_data.d_blendingMask, m_data.d_occlusionMask, sizeof(float) * rayCastParams.m_height * rayCastParams.m_width, cudaMemcpyDeviceToDevice);
	
		//maskSlantRegionCUDA(rayCastData, depthCameraData, rayCastParams, m_data.d_blendingMask);

		erodeMask(m_data.d_blendingMask, m_params.m_erode_iter_stretch_box);
		erodeMaskSoft(m_data.d_blendingMask, m_params.m_erode_iter_stretch_gauss, m_params.m_erode_sigma_stretch);

#ifdef SHOWMASK
		displayFloatMat(m_data.d_blendingMask, "blending Mask", rayCastParams.m_width, rayCastParams.m_height);
#endif
	}

	void computeMasks(const RayCastData &rayCastData, const DepthCameraData &depthCameraData, const RayCastParams &rayCastParams) {

		// detect occlusion edges of the object and erode it
		computeOcclusionMask(rayCastData, depthCameraData, rayCastParams); 

		// mask out region observed in a grazing angle
		computeSourceMask(rayCastData, depthCameraData, rayCastParams);

		// mask for image ( no mask )
		computeTargetMask(rayCastData, depthCameraData, rayCastParams); // depth invalid part 

		// compute a warping mask
		computeWarpingMask(rayCastData, depthCameraData, rayCastParams);
		
		// compute a blending mask 
		computeBlendingMask(rayCastData, depthCameraData, rayCastParams); // occlusion region + screen then soft erode.

#ifdef SHOWMASK
		cv::waitKey(0);
#endif
	}

	const TexUpdateData& getTexUpdateData(void) {
		return m_data;
	}

	const TexUpdateParams& getTexUpdateParams() const {
		return m_params;
	}

	TexPoolData& getTexPoolData() {
		return m_texPoolData;
	}

	const TexPoolParams& getTexPoolParams() const {
		return m_texPoolParams;
	}

	//used
	void findZeroCrossingVoxels(HashData& hashData, HashParams& hashParams, const DepthCameraData& depthCameraData) {

		//for every voxel, check whether this voxel is zero-crossing or not.
		hashParams.m_numZeroCrossVoxels = findZeroCrossingVoxelsCUDA(hashData, hashParams, m_texPoolData, depthCameraData); //should update Hash param after call

	}

	void deletePreviousTexTriangle(HashData& hashData, HashParams& hashParams, const DepthCameraData& depthCameraData) {

		deletePreviousTexTriangleCUDA(hashData, hashParams, m_texPoolData, depthCameraData); //should update Hash param after call
				
	}

	//used
	void computeTexDepth(HashData& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams) {
	
		computeTexDepthCUDA(hashData, hashParams, m_texPoolData, m_texPoolParams, m_rayCastData, depthCameraData);
	
	}

	//
	void texUpdateFromPreviousTexture(HashData& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams) {
		
		texUpdateFromPreviousTextureCUDA(hashData, hashParams, m_texPoolData, m_texPoolParams, m_rayCastData, depthCameraData);
	}

	//
	void texUpdateFromImage(HashData& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams) {
		texUpdateFromImageCUDA(hashData, hashParams, m_data, m_texPoolData, m_texPoolParams, m_rayCastData, depthCameraData, m_data.d_blendingMask);
	}

	void texUpdateFromImageHalf(HashData& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams) {
		texUpdateFromImageHalfCUDA(hashData, hashParams, m_data, m_texPoolData, m_texPoolParams, m_rayCastData, depthCameraData, m_data.d_blendingMask);
	}
	
	void texUpdateFromImageWithCameraMotionMap(HashData& hashData, const HashParams& hashParams,
			const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, float *d_motionmap){
		//texUpdateFromImageWithCameraMotionMapCUDA

		texUpdateFromImageWithCameraMotionMapCUDA(hashData, hashParams, m_data, m_texPoolData, m_texPoolParams, m_rayCastData, depthCameraData, d_motionmap, m_data.d_blendingMask);
		
	}

	void texUpdateFromImageWithCameraMotionMapHalf(HashData& hashData, const HashParams& hashParams,
		const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, float *d_motionmap) {
		//texUpdateFromImageWithCameraMotionMapCUDA

		texUpdateFromImageWithCameraMotionMapHalfCUDA(hashData, hashParams, m_data, m_texPoolData, m_texPoolParams, m_rayCastData, depthCameraData, d_motionmap, m_data.d_blendingMask);

	}
	
	void texUpdateFromImageOnly(HashData& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams) {
		texUpdateFromImageOnlyCUDA(hashData, hashParams, m_data, m_texPoolData, m_texPoolParams, m_rayCastData, depthCameraData, m_data.d_blendingMask);
	}

	void resetColor(HashData& hashData, const HashParams& hashParams) {
		
		resetColorCUDA(hashData, hashParams, m_texPoolData, m_texPoolParams);


	}

	void updateMasksWithCapturedFrame(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, const float4 *d_normalmap_sensor ) {

		updateMasksWithCapturedFrameCUDA(depthCameraData, depthCameraParams, d_normalmap_sensor, m_data);
	
		erodeMask(m_data.d_depthMask, m_params.m_erode_iter_occdepth);

#ifdef SHOWMASK
		displayFloatMat(m_data.d_depthMask, "depth Mask", depthCameraParams.m_imageWidth, depthCameraParams.m_imageHeight);
#endif
	}

private:

	void create(const TexUpdateParams& params);
	void create(const TexUpdateParams& params, const TexPoolParams& texParams, const RayCastParams& rayCastParams) {

		m_params = params;
		m_data.allocate(m_params);

		//tex Pool
		m_texPoolParams = texParams;
		m_texPoolData.allocate(m_texPoolParams);

		m_rayCastParams = rayCastParams;

		reset();
	}
	
	void reset() {

		m_data.updateParams(m_params);

		m_texPoolData.updateParams(m_texPoolParams);
		resetTexCUDA(m_texPoolData, m_texPoolParams);

		m_rayCastData.updateParams(m_rayCastParams);

	}

	void destroy(void);
		
	TexUpdateParams m_params;
	TexUpdateData m_data;

	TexPoolParams	m_texPoolParams;
	TexPoolData		m_texPoolData;

	RayCastParams	m_rayCastParams;
	RayCastData		m_rayCastData;

	static Timer m_timer;

};