#include "CUDAImageUtil.h"

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#define T_PER_BLOCK 32
#define KERNEL_MAX_WIDTH 5

__device__ __constant__ float d_kernel[10][KERNEL_MAX_WIDTH];

inline __device__ float gaussR(float sigma, float dist)
{
	return exp(-(dist*dist) / (2.0*sigma*sigma));
}

extern inline __device__ float linearR(float sigma, float dist)
{
	return max(1.0f, min(0.0f, 1.0f - (dist*dist) / (2.0*sigma*sigma)));
}

extern inline __device__ float gaussD(float sigma, int x, int y)
{
	return exp(-((x*x + y*y) / (2.0f*sigma*sigma)));
}

extern inline __device__ float gaussD(float sigma, int x)
{
	return exp(-((x*x) / (2.0f*sigma*sigma)));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//horizontal filter block size (32, 64, 128, 256, 512)
#define FILTERH_TILE_WIDTH 128

//thread block for vertical filter. FILTERV_BLOCK_WIDTH can be (4, 8 or 16)
#define FILTERV_BLOCK_WIDTH 16
#define FILTERV_BLOCK_HEIGHT 32

//The corresponding image patch for a thread block
#define FILTERV_PIXEL_PER_THREAD 4
#define FILTERV_TILE_WIDTH FILTERV_BLOCK_WIDTH
#define FILTERV_TILE_HEIGHT (FILTERV_PIXEL_PER_THREAD * FILTERV_BLOCK_HEIGHT)

#define IMUL(X,Y) __mul24(X,Y)

template<int FW> __global__ void FilterH(float* d_input, float* d_result, int width, unsigned int filterIndex)
{

	const int HALF_WIDTH = FW >> 1;
	const int CACHE_WIDTH = FILTERH_TILE_WIDTH + FW - 1;
	const int CACHE_COUNT = 2 + (CACHE_WIDTH - 2) / FILTERH_TILE_WIDTH;
	__shared__ float data[CACHE_WIDTH];
	const int bcol = IMUL(blockIdx.x, FILTERH_TILE_WIDTH);
	const int col = bcol + threadIdx.x;
	const int index_min = IMUL(blockIdx.y, width);
	const int index_max = index_min + width - 1;
	int src_index = index_min + bcol - HALF_WIDTH + threadIdx.x;
	int cache_index = threadIdx.x;
	float value = 0;
#pragma unroll
	for (int j = 0; j < CACHE_COUNT; ++j)
	{
		if (cache_index < CACHE_WIDTH)
		{
			int fetch_index = src_index < index_min ? index_min : (src_index > index_max ? index_max : src_index);
			data[cache_index] = d_input[ fetch_index];
			src_index += FILTERH_TILE_WIDTH;
			cache_index += FILTERH_TILE_WIDTH;
		}
	}
	__syncthreads();
	if (col >= width) return;
#pragma unroll
	for (int i = 0; i < FW; ++i)
	{
		value += (data[threadIdx.x + i] * d_kernel[filterIndex][i]);
	}
	//	value = Conv<FW-1>(data + threadIdx.x);
	d_result[index_min + col] = value;
}



////////////////////////////////////////////////////////////////////
template<int  FW>  __global__ void FilterV(float* d_input,float* d_result, int width, int height, unsigned int filterIndex)
{
	const int HALF_WIDTH = FW >> 1;
	const int CACHE_WIDTH = FW + FILTERV_TILE_HEIGHT - 1;
	const int TEMP = CACHE_WIDTH & 0xf;
	//add some extra space to avoid bank conflict
#if FILTERV_TILE_WIDTH == 16
	//make the stride 16 * n +/- 1
	const int EXTRA = (TEMP == 1 || TEMP == 0) ? 1 - TEMP : 15 - TEMP;
#elif FILTERV_TILE_WIDTH == 8
	//make the stride 16 * n +/- 2
	const int EXTRA = (TEMP == 2 || TEMP == 1 || TEMP == 0) ? 2 - TEMP : (TEMP == 15 ? 3 : 14 - TEMP);
#elif FILTERV_TILE_WIDTH == 4
	//make the stride 16 * n +/- 4
	const int EXTRA = (TEMP >= 0 && TEMP <= 4) ? 4 - TEMP : (TEMP > 12 ? 20 - TEMP : 12 - TEMP);
#else
#error
#endif
	const int CACHE_TRUE_WIDTH = CACHE_WIDTH + EXTRA;
	const int CACHE_COUNT = (CACHE_WIDTH + FILTERV_BLOCK_HEIGHT - 1) / FILTERV_BLOCK_HEIGHT;
	const int WRITE_COUNT = (FILTERV_TILE_HEIGHT + FILTERV_BLOCK_HEIGHT - 1) / FILTERV_BLOCK_HEIGHT;
	__shared__ float data[CACHE_TRUE_WIDTH * FILTERV_TILE_WIDTH];
	const int row_block_first = IMUL(blockIdx.y, FILTERV_TILE_HEIGHT);
	const int col = IMUL(blockIdx.x, FILTERV_TILE_WIDTH) + threadIdx.x;
	const int row_first = row_block_first - HALF_WIDTH;
	const int data_index_max = IMUL(height - 1, width) + col;
	const int cache_col_start = threadIdx.y;
	const int cache_row_start = IMUL(threadIdx.x, CACHE_TRUE_WIDTH);
	int cache_index = cache_col_start + cache_row_start;
	int data_index = IMUL(row_first + cache_col_start, width) + col;

	if (col < width)
	{
#pragma unroll
		for (int i = 0; i < CACHE_COUNT; ++i)
		{
			if (cache_col_start < CACHE_WIDTH - i * FILTERV_BLOCK_HEIGHT)
			{
				int fetch_index = data_index < col ? col : (data_index > data_index_max ? data_index_max : data_index);
				data[cache_index + i * FILTERV_BLOCK_HEIGHT] = d_input[fetch_index];
				data_index += IMUL(FILTERV_BLOCK_HEIGHT, width);
			}
		}
	}
	__syncthreads();

	if (col >= width) return;

	int row = row_block_first + threadIdx.y;
	int index_start = cache_row_start + threadIdx.y;
#pragma unroll
	for (int i = 0; i < WRITE_COUNT;		++i,
		row += FILTERV_BLOCK_HEIGHT, index_start += FILTERV_BLOCK_HEIGHT)
	{
		if (row < height)
		{
			int index_dest = IMUL(row, width) + col;
			float value = 0;
#pragma unroll
			for (int i = 0; i < FW; ++i)
			{
				value += (data[index_start + i] * d_kernel[filterIndex][i]);
			}
			d_result[index_dest] = value;
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<int FW> void filterImage(float *dst, float *src, float* buf, int width, int height, unsigned int filterIndex)
{

	//horizontal filtering
	dim3 gridh((width + FILTERH_TILE_WIDTH - 1) / FILTERH_TILE_WIDTH, height);
	dim3 blockh(FILTERH_TILE_WIDTH);
	FilterH<FW> << <gridh, blockh >> >(src, buf, width, filterIndex);
//	CheckErrorCUDA("FilterH");

	///vertical filtering
//	buf->BindTexture(texData);
	dim3 gridv((width + FILTERV_TILE_WIDTH - 1) / FILTERV_TILE_WIDTH, (height + FILTERV_TILE_HEIGHT - 1) / FILTERV_TILE_HEIGHT);
	dim3 blockv(FILTERV_TILE_WIDTH, FILTERV_BLOCK_HEIGHT);
	FilterV<FW> << <gridv, blockv >> >(buf, dst, width, height, filterIndex);
//	CheckErrorCUDA("FilterV");
}

//////////////////////////////////////////////////////////////////////
// tested on 2048x1500 image, the time on pyramid construction is
// OpenGL version : 18ms
// CUDA version: 28 ms
void filterImage(float *dst, float *src, float* buf, unsigned int width, unsigned int height, unsigned int filterIndex)
{
	//CUDATimer timer;
	//timer.startEvent("FilterImage");

	switch (width)
	{
	case 3:		filterImage< 3>(dst, src, buf, width, height, filterIndex);	break;
	case 5:		filterImage< 5>(dst, src, buf, width, height, filterIndex);	break;
	case 7:		filterImage< 7>(dst, src, buf, width, height, filterIndex);	break;
	case 9:		filterImage< 9>(dst, src, buf, width, height, filterIndex);	break;
	case 11:	filterImage<11>(dst, src, buf, width, height, filterIndex);	break;
	case 13:	filterImage<13>(dst, src, buf, width, height, filterIndex);	break;
	case 15:	filterImage<15>(dst, src, buf, width, height, filterIndex);	break;
	case 17:	filterImage<17>(dst, src, buf, width, height, filterIndex);	break;
	case 19:	filterImage<19>(dst, src, buf, width, height, filterIndex);	break;
	case 21:	filterImage<21>(dst, src, buf, width, height, filterIndex);	break;
	case 23:	filterImage<23>(dst, src, buf, width, height, filterIndex);	break;
	case 25:	filterImage<25>(dst, src, buf, width, height, filterIndex);	break;
	case 27:	filterImage<27>(dst, src, buf, width, height, filterIndex);	break;
	case 29:	filterImage<29>(dst, src, buf, width, height, filterIndex);	break;
	case 31:	filterImage<31>(dst, src, buf, width, height, filterIndex);	break;
	case 33:	filterImage<33>(dst, src, buf, width, height, filterIndex);	break;
	default:	break;
	}
	//timer.endEvent();
	//if (src->GetImgWidth() == 1296 && width  == 25) timer.evaluate();
}

void initFilterKernels()
{
	float kernel[KERNEL_MAX_WIDTH];

	//0. Gaussian kernel
	//1. Gradient kernel

	
	int index;

	memset(kernel, 0, sizeof(float) * KERNEL_MAX_WIDTH);
	index = 0;
	kernel[0] = 1.f / 4.f;
	kernel[1] = 2.f / 4.f;
	kernel[2] = 1.f / 4.f;

	cudaMemcpyToSymbol(d_kernel, kernel, KERNEL_MAX_WIDTH * sizeof(float), index * KERNEL_MAX_WIDTH * sizeof(float));

	memset(kernel, 0, sizeof(float) * KERNEL_MAX_WIDTH);
	index = 1;
	kernel[0] = -0.5f;
	kernel[1] = 0.f;
	kernel[2] = 0.5f;

	cudaMemcpyToSymbol(d_kernel, kernel, KERNEL_MAX_WIDTH * sizeof(float), index * KERNEL_MAX_WIDTH * sizeof(float));

	
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void computeSobelGradientKernel(float2 * d_output, float *d_input, unsigned int width, unsigned int height) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = y * width + x;
	float dx, dy;
	float value;
	dx = dy = 0;
	if (0 < x && x < width-1 && 0 < y && y < height-1) {

		dx = -0.125 * (d_input[index - width - 1] + 2 * d_input[index - 1] + d_input[index + width - 1])
			+ 0.125 *(d_input[index - width + 1] + 2 * d_input[index + 1] + d_input[index + width + 1]);
		dy = -0.125 * (d_input[index - width - 1] +  2 * d_input[index - width] + d_input[index - width + 1])
			+ 0.125 * (d_input[index + width - 1] + 2 * d_input[index +width ] + d_input[index + width + 1]);
		d_output[index] = make_float2(dx, dy);
	}

}

extern "C" void computeSobelGradient(float2 * d_output, float *d_input, unsigned int width, unsigned int height) {

	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeSobelGradientKernel << <gridSize, blockSize >> >(d_output, d_input, width, height);
	
}

__global__ void computeGradient3Kernel(float2 * d_output, float *d_input, unsigned int width, unsigned int height) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = y * width + x;
	float dx, dy;
	float value;
	dx = dy = 0;
	if (0 < x && x < width - 1 && 0 < y && y < height - 1) {

		dx = -0.5 * (d_input[index - 1])
			+ 0.5 *(d_input[index + 1]);
		dy = -0.5 * (d_input[index - width])
			+ 0.5 * (d_input[index + width]);
		d_output[index] = make_float2(dx, dy);
	}

}

extern "C" void computeGradient3(float2 * d_output, float *d_input, unsigned int width, unsigned int height) {

	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeGradient3Kernel << <gridSize, blockSize >> >(d_output, d_input, width, height);

}

__global__ void computeGradient2Kernel(float2 * d_output, float *d_input, unsigned int width, unsigned int height) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = y * width + x;
	float dx, dy;
	float value;
	dx = dy = 0;
	if (0 < x && x < width - 1 && 0 < y && y < height - 1) {

		dx = - (d_input[index ])
			+ (d_input[index + 1]);
		dy = - (d_input[index ])
			+ (d_input[index + width]);
		d_output[index] = make_float2(dx, dy);
	}

}

extern "C" void computeGradient2(float2 * d_output, float *d_input, unsigned int width, unsigned int height) {

	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeGradient2Kernel << <gridSize, blockSize >> >(d_output, d_input, width, height);

}



__global__ void gaussFilterAlphaMapDevice(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	const int kernelRadius = (int)ceil(2.0*sigmaD);

	d_output[y*width + x] = 0;

	float sum = 0.0f;
	float sumWeight = 0.0f;

	const float valueCenter = d_input[y*width + x];
	d_output[y*width + x] = valueCenter;
	if (valueCenter < sigmaR - 0.001)
	{
		for (int m = x - kernelRadius; m <= x + kernelRadius; m++)
		{
			for (int n = y - kernelRadius; n <= y + kernelRadius; n++)
			{
				if (m >= 0 && n >= 0 && m < width && n < height)
				{
					const float currentValue = d_input[n*width + m];

					const float weight = gaussD(sigmaD, m - x, n - y);

					sumWeight += weight;
					sum += weight*currentValue;
				}
			}
		}
		if (sumWeight > 0.0f) d_output[y*width + x] = sum / sumWeight;
	}
}

extern "C" void gaussFilterAlphaMap(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	gaussFilterAlphaMapDevice << <gridSize, blockSize >> > (d_output, d_input, sigmaD, sigmaR, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void gaussBlur_kernel(float* d_output, float* d_input, float *d_mask, float sigmaD, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	const int kernelRadius = (int)ceil(2.0*sigmaD);

	d_output[y*width + x] = 0;

	float sum = 0.0f;
	float sumWeight = 0.0f;

	const float valueCenter = d_input[y*width + x];
	d_output[y*width + x] = valueCenter;

	for (int m = x - kernelRadius; m <= x + kernelRadius; m++)
	{
		for (int n = y - kernelRadius; n <= y + kernelRadius; n++)
		{
			if (m >= 0 && n >= 0 && m < width && n < height)
			{
				const float currentValue = d_input[n*width + m];
				const float maskValue = d_mask[n*width + m];
				
				if (maskValue <0.1) {

					const float weight = gaussD(sigmaD, m - x, n - y);
					sumWeight += weight;
					sum += weight*currentValue;
				
				}
			}
		}
	}

	if (sumWeight > 0.0f) d_output[y*width + x] = sum / sumWeight;

}

__global__ void gaussBlurWOMask_kernel(float* d_output, float* d_input, float sigmaD, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	const int kernelRadius = (int)ceil(2.0*sigmaD);

	d_output[y*width + x] = 0;

	float sum = 0.0f;
	float sumWeight = 0.0f;

	const float valueCenter = d_input[y*width + x];
	d_output[y*width + x] = valueCenter;

	for (int m = x - kernelRadius; m <= x + kernelRadius; m++)
	{
		for (int n = y - kernelRadius; n <= y + kernelRadius; n++)
		{
			if (m >= 0 && n >= 0 && m < width && n < height)
			{
				const float currentValue = d_input[n*width + m];

				

					const float weight = gaussD(sigmaD, m - x, n - y);
					sumWeight += weight;
					sum += weight*currentValue;
				
			}
		}
	}

	if (sumWeight > 0.0f) d_output[y*width + x] = sum / sumWeight;

}

extern "C" void gaussBlur(float* d_output, float* d_input, float *d_mask, float sigmaD, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	gaussBlur_kernel << <gridSize, blockSize >> >(d_output, d_input, d_mask, sigmaD, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

extern "C" void gaussBlurWOMask(float* d_output, float* d_input, float sigmaD, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	gaussBlurWOMask_kernel << <gridSize, blockSize >> >(d_output, d_input, sigmaD, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void downSample_kernel(float* d_output, float* d_input, unsigned int width, unsigned int height)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int hx = x << 1;
	const int hy = y << 1;
	const int hw = width << 1; 

	if (x >= width || y >= height) return;

	d_output[y * width + x] = 0.25 * (d_input[hy * hw + hx] + d_input[hy * hw + hx + 1] + d_input[hy * hw + hx + hw] + d_input[hy * hw + hx + hw + 1]);
//	d_output[y * width + x] =d_input[hy * hw + hx] ;

}

extern "C" void downSample(float* d_output, float* d_input, unsigned int width, unsigned int height){

	int lwidth = width>> 1;
	int lheight = height >> 1;

	const dim3 gridSize((lwidth + T_PER_BLOCK - 1) / T_PER_BLOCK, (lheight + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	downSample_kernel <<<gridSize, blockSize>>> (d_output, d_input, lwidth, lheight);

}

//do we use this function for what?
__global__ void downSampleMask_kernel(float* d_output, float* d_input, unsigned int width, unsigned int height)
{

	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int hx = x << 1;
	const int hy = y << 1;
	const int hw = width << 1; 
	const int hh = height << 1;

	if (x >= width || y >= height) return;

	float value = -1.;

	for (int offx = -1; offx <= 1; offx++) {

		for (int offy = -1; offy <= 1; offy++) {

			int ix = hx + offx;
			int iy = hy + offy;
			
			if (0 <= ix && 0 <= iy && ix < hw && iy < hh) {

				if (value < d_input[iy*hw + ix])
					value = d_input[iy*hw + ix];
			
			}
		}
	}

	d_output[ y * width + x ] = value;

}

extern "C" void downSampleMask(float* d_output, float* d_input, unsigned int width, unsigned int height){

	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	downSampleMask_kernel <<<gridSize, blockSize>>> (d_output, d_input, width, height);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Color to Gray scale image convert
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertColor2GrayDevice(float* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	d_output[y*width + x] = 0.0f;
	const float4 color_value = d_input[y*width + x];
	float gray_value = 0.299f * color_value.x + 0.587f * color_value.y + 0.114f * color_value.z;
	if (gray_value < 0.f) gray_value = 0.0f;
	if (gray_value > 1.0f) gray_value = 0.0f;

	d_output[y*width + x] = gray_value;
}


extern "C" void convertColor2Gray(float* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	convertColor2GrayDevice << <gridSize, blockSize >> > (d_output, d_input, width, height);
}

__global__ void opticalFlowToColorDevice(float2* flow, float4* flow_colorization, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	flow_colorization[y * width + x] = make_float4(make_float3(0.0f), 1.0f);

	//if (flow[y * width + x].x != 0.0f) flow_colorization[y * width + x].x = 255.0f;
	flow_colorization[y * width + x].x = abs(flow[y * width + x].x);
	flow_colorization[y * width + x].y = abs(flow[y * width + x].y);
}

extern "C" void opticalFlowToColor(float2 *d_flow, float4* d_flow_colorization, unsigned int width, unsigned int height)
{

	const dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK, (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	opticalFlowToColorDevice << <gridSize, blockSize >> > (d_flow, d_flow_colorization, width, height);

}