
#include "CUDAHashParams.h"
#include "CUDARayCastParams.h"
#include "CUDADepthCameraParams.h"
#include "CUDATexPoolParams.h"
#include "CUDATexUpdateParams.h"

__constant__ HashParams c_hashParams;
__constant__ RayCastParams c_rayCastParams;
__constant__ DepthCameraParams c_depthCameraParams;
__constant__ TexPoolParams c_texPoolParams;
__constant__ TexUpdateParams c_texUpdateParams;

extern "C" void updateConstantTexUpdateParams(const TexUpdateParams& params) {

	size_t size;
	cutilSafeCall(cudaGetSymbolSize(&size, c_texUpdateParams));
	cutilSafeCall(cudaMemcpyToSymbol(c_texUpdateParams, &params, size, 0, cudaMemcpyHostToDevice));

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


extern "C" void updateConstantTexPoolParams(const TexPoolParams& params) {

	size_t size;
	cutilSafeCall(cudaGetSymbolSize(&size, c_texPoolParams));
	cutilSafeCall(cudaMemcpyToSymbol(c_texPoolParams, &params, size, 0, cudaMemcpyHostToDevice));

#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

extern "C" void updateConstantHashParams(const HashParams& params) {

	size_t size;
	cutilSafeCall(cudaGetSymbolSize(&size, c_hashParams));
	cutilSafeCall(cudaMemcpyToSymbol(c_hashParams, &params, size, 0, cudaMemcpyHostToDevice));
	
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	}


extern "C" void updateConstantRayCastParams(const RayCastParams& params) {
	
	size_t size;
	cutilSafeCall(cudaGetSymbolSize(&size, c_rayCastParams));
	cutilSafeCall(cudaMemcpyToSymbol(c_rayCastParams, &params, size, 0, cudaMemcpyHostToDevice));
	
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

}

extern "C" void updateConstantDepthCameraParams(const DepthCameraParams& params) {
	
	size_t size;
	cutilSafeCall(cudaGetSymbolSize(&size, c_depthCameraParams));
	cutilSafeCall(cudaMemcpyToSymbol(c_depthCameraParams, &params, size, 0, cudaMemcpyHostToDevice));
	
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

}

