#include "stdafx.h"

#include "texturePool.h"
#include "VoxelUtilHashSDF.h"
#include "RayCastSDFUtil.h"
#include "TexUpdateUtil.h"

#include "Util.h"

#include "CUDATexUpdate.h"

#include "cudaDebug.h"

Timer CUDATexUpdate::m_timer;

extern "C" void gaussBlurWOMask(float* d_output, float* d_input, float sigmaD, unsigned int width, unsigned int height);
extern "C" void gaussFilterAlphaMap(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height);
extern "C" void erodeHole(float* d_output, float* d_input, int raidus, unsigned int width, unsigned int height);
extern "C" void erodeDistCUDA(float* d_output, float* d_input, int raidus, float degradeFactor,  unsigned int width, unsigned int height);
extern "C" void setAlpha(float* d_output, float* d_input, unsigned int width, unsigned int height);
extern "C" void convertColor2Gray(float* d_output, float4* d_input, unsigned int width, unsigned int height);
extern "C" void opticalFlowToColor(float2 *d_flow, float4* d_flow_colorization, unsigned int width, unsigned int height);
extern "C" void exportTextureImage(uchar *d_output, uint *d_heap, Texel *d_input, unsigned int tex_width, unsigned int tex_height, unsigned int patch_width, unsigned int patch_size, unsigned int num_patches, unsigned int max_num_patches);

void CUDATexUpdate::create(const TexUpdateParams& params)
{
	m_params = params;
	m_data.allocate(m_params);
}

void CUDATexUpdate::destroy(void)
{
	m_data.free();
;}

void CUDATexUpdate::computeAlpha(int iter_n, float sigma) {

//	setAlpha(m_data.d_alpha, d_depth, m_params.m_width, m_params.m_height);

	int even_iter = (iter_n >> 1) << 1;
	for (int i = 0; i < even_iter; i++) {

		if (i & 1) {
			gaussFilterAlphaMap(m_data.d_alpha, m_data.d_alphaHelper, sigma, 1.0f, m_params.m_width, m_params.m_height);
		}
		else {
			gaussFilterAlphaMap(m_data.d_alphaHelper, m_data.d_alpha, sigma, 1.0f, m_params.m_width, m_params.m_height);
			//displayFloatMat((float*)m_data.d_alpha, "masks", m_params.m_width, m_params.m_height);
			//cv::waitKey(0);
		}

	}

}

//used
void CUDATexUpdate::erodeMask(float *d_mask, int erode_iter) {

	int even_iter = (erode_iter >> 1) << 1;
	int radius = 1;
	for (int i = 0; i < even_iter; i++) {

		if (i & 1) {
			erodeHole(d_mask, m_data.d_alphaHelper, radius, m_params.m_width, m_params.m_height);
		}
		else {
			erodeHole(m_data.d_alphaHelper, d_mask, radius, m_params.m_width, m_params.m_height);
			
		}
	}
}

void CUDATexUpdate::erodeDist(float *d_mask, int erode_iter) {

	int even_iter = (erode_iter >> 1) << 1;
	int radius = 2;
	float degradeFactor = 1.f / (float)erode_iter;
	printf("degradefactor %f %d\n", degradeFactor, erode_iter);
	for (int i = 0; i < even_iter; i++) {

		if (i & 1) {
			erodeDistCUDA(d_mask, m_data.d_alphaHelper, radius, degradeFactor, m_params.m_width, m_params.m_height);
		}
		else {
			erodeDistCUDA(m_data.d_alphaHelper, d_mask, radius, degradeFactor, m_params.m_width, m_params.m_height);

		}
	}
}

void CUDATexUpdate::erodeMaskSoft(float *d_mask, int erode_iter, float sigma) {

	int even_iter = (erode_iter >> 1) << 1;
	for (int i = 0; i < even_iter; i++) {

		if (i & 1) {
			gaussBlurWOMask(d_mask, m_data.d_alphaHelper, sigma, m_params.m_width, m_params.m_height);
		}
		else {

			gaussBlurWOMask(m_data.d_alphaHelper, d_mask,  sigma, m_params.m_width, m_params.m_height);

		}
	}
}

void CUDATexUpdate::extractTexture() {
	int maxTextureWidth = 256;
	uchar *d_textureImg;
	uchar *h_textureImg = (uchar *)malloc(sizeof(uchar) * 3 * m_texPoolParams.m_texturePatchSize * maxTextureWidth * maxTextureWidth);
	
	cutilSafeCall(cudaMalloc(&d_textureImg, sizeof(uchar) * 3 * m_texPoolParams.m_texturePatchSize * maxTextureWidth * maxTextureWidth));
	cudaMemset(d_textureImg, 0, sizeof(uchar) * 3 * m_texPoolParams.m_texturePatchSize * maxTextureWidth * maxTextureWidth);
	//Here we save our texture.
	uint h_heapCounter;
	cudaMemcpy(&h_heapCounter, m_texPoolData.d_heapCounter, sizeof(uint), cudaMemcpyDeviceToHost);
	uint h_numTextureTile = m_texPoolParams.m_numTexturePatches - h_heapCounter;
	uint textureFullRes = h_numTextureTile * m_texPoolParams.m_texturePatchSize;
	

	uint exportTextureWidth = std::sqrt(h_numTextureTile);
	std::cout << "Number of texture: " << h_numTextureTile << std::endl;
	std::cout << "Texture Width: " << exportTextureWidth << std::endl;
	std::cout << "Texture full res: " << textureFullRes << std::endl;

	exportTextureImage(d_textureImg, m_texPoolData.d_heap, m_texPoolData.d_texPatches, maxTextureWidth, maxTextureWidth, m_texPoolParams.m_texturePatchWidth, m_texPoolParams.m_texturePatchSize, h_numTextureTile, m_texPoolParams.m_numTexturePatches);
	cudaMemcpy(h_textureImg, d_textureImg, sizeof(uchar) * 3 * m_texPoolParams.m_texturePatchSize * maxTextureWidth * maxTextureWidth, cudaMemcpyDeviceToHost);

	unsigned int size = maxTextureWidth * m_texPoolParams.m_texturePatchWidth * maxTextureWidth * m_texPoolParams.m_texturePatchWidth;
	vec3f *data = new vec3f[size];
	for (unsigned int i = 0; i < size; i++) {
		data[i] = vec3f(h_textureImg[i]) / 255.f;
	}
	cv::Mat temp(maxTextureWidth *  m_texPoolParams.m_texturePatchWidth, maxTextureWidth *  m_texPoolParams.m_texturePatchWidth, CV_8UC3, h_textureImg);
	cv::cvtColor(temp, temp, cv::COLOR_RGB2BGR);
	cv::imshow("texture", temp);
	cv::imwrite("Scans/texture.png", temp);

	SAFE_DELETE_ARRAY(data);
	SAFE_DELETE_ARRAY(h_textureImg);
	cutilSafeCall(cudaFree(d_textureImg));
}

template <typename T>
inline T mapVal(T x, T a, T b, T c, T d)
{
	x = ::max(::min(x, b), a);
	return c + (d - c) * (x - a) / (b - a);
}
