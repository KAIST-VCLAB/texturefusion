#ifndef __CUDADEBUG_H__
#define __CUDADEBUG_H__

#include <iostream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>

template<typename T>
static void CheckRuntimeError(T error, const std::string& functionName, const std::string& fileName, const int line, const char* message)
{
	if (error != cudaSuccess)
	{
		std::cerr << "CUDA ERROR:" << std::endl;
		std::cerr << "   file: " << fileName << std::endl;
		std::cerr << "   line: " << line << std::endl;
		std::cerr << "   func: " << functionName << std::endl;
		std::cerr << "   msg: " << message << std::endl;
		std::cerr << "   desc: " << cudaGetErrorString(error) << std::endl;
		throw std::runtime_error(message);

	}
}
#define CHECK_CUDA_ERROR(error, functionName, message) CheckRuntimeError( (error), functionName, __FILE__, __LINE__, (message) );

extern "C"
static void printFloat4x4(float4x4 a, std::string name){
	printf("%s\n", name.c_str());
	for(int i=0;i<4;i++){
		for(int j=0;j<4;j++){
			printf("%f ", a(i,j));
		}
		printf("\n");
	}
}

static bool writeUint3Array(uint3 *d_array, std::string filename, int n) {

	printf("write uint3 array\n");

	if (d_array != NULL) {

		FILE *out = fopen(filename.data(), "w");
		uint3 *h_array = (uint3*)malloc(sizeof(uint3) * n);

		cudaMemcpy(h_array, d_array, sizeof(uint3) * n, cudaMemcpyDeviceToHost);

		int i;

		printf("write uint3 array\n");

		//print the number of vectors
		fprintf(out, "%d\n", n);

		printf("write uint3 array\n");

		//print triangle indices
		for (i = 0; i<n; i++)
			fprintf(out, "%u %u %u \n ", h_array[i].x, h_array[i].y, h_array[i].z);

		printf("finished\n");

		free(h_array);

		return true;

	}
	else return false;
}

static bool writeUint3ArrayHost(uint3 *h_array, std::string filename, int n) {

	printf("write uint3 array\n");

	if (h_array != NULL) {

		FILE *out = fopen(filename.data(), "w");

		int i;

		printf("write uint3 array\n");

		//print the number of vectors
		fprintf(out, "%d\n", n);

		printf("write uint3 array\n");

		//print triangle indices
		for (i = 0; i<n; i++)
			fprintf(out, "%u %u %u \n ", h_array[i].x, h_array[i].y, h_array[i].z);

		printf("finished\n");

		return true;

	}
	else return false;
}

static bool writeFloat3Array(float3 *d_array, std::string filename, int n) {

	printf("write uint3 array\n");

	if (d_array != NULL) {

		FILE *out = fopen(filename.data(), "w");
		float3 *h_array = (float3*)malloc(sizeof(float3) * n);

		cudaMemcpy(h_array, d_array, sizeof(float3) * n, cudaMemcpyDeviceToHost);

		int i;

		printf("write float3 array\n");

		//print the number of vectors
		fprintf(out, "%d\n", n);

		printf("write float3 array\n");

		//print triangle indices
		for (i = 0; i<n; i++)
			fprintf(out, "%f %f %f \n ", h_array[i].x, h_array[i].y, h_array[i].z);

		printf("finished\n");

		free(h_array);


		return true;

	}
	else return false;
}
static bool writeFloatArray(float *d_array, std::string filename, int n) {

	printf("write float array\n");

	if (d_array != NULL) {

		FILE *out = fopen(filename.data(), "w");
		float *h_array = (float*)malloc(sizeof(float) * n);

		cudaMemcpy(h_array, d_array, sizeof(float) * n, cudaMemcpyDeviceToHost);

		int i;
    
		//print the number of vectors
		fprintf(out, "%d\n", n);

		//print triangle indices
		for (i = 0; i<n; i++)
			fprintf(out, "%f \n ", h_array[i]);

		printf("finished\n");

		free(h_array);

		fclose(out);

		return true;

	}
	else return false;
}



static bool writeFloatNArray(float *d_array, std::string filename, int n, int ch) {

  if (d_array != NULL) {

    FILE *out = fopen(filename.data(), "w");
    float *h_array = (float*)malloc(sizeof(float) * n * ch);

    cudaMemcpy(h_array, d_array, sizeof(float) * n *ch, cudaMemcpyDeviceToHost);

    int i;

    printf("write float array\n");

    //print the number of vectors
    fprintf(out, "%d\n", n);

    //print triangle indices
    for (i = 0; i < n; i++) {
      for (int j = 0; j < ch; j++)
        if(h_array[i*ch + j]>0.0001)
          fprintf(out, "%.4f  ", h_array[i*ch + j]);
      fprintf(out, "\n");
    }
    printf("finished\n");

    free(h_array);

    return true;

  }
  else return false;
}
static bool writeFloat3ArrayHost(float3 *h_array, std::string filename, int n) {

	if (h_array != NULL) {

		FILE *out = fopen(filename.data(), "w");

		int i;

		printf("write float3 array\n");

		//print the number of vectors
		fprintf(out, "%d\n", n);

		printf("write float3 array\n");

		//print triangle indices
		for (i = 0; i<n; i++)
			fprintf(out, "%f %f %f \n ", h_array[i].x, h_array[i].y, h_array[i].z);

		printf("finished\n");

		return true;

	}
	else return false;
}

static bool writeUintArray(unsigned int *d_array, std::string filename, int n) {

	if (d_array != NULL) {

		FILE *out = fopen(filename.data(), "w");
		unsigned int *h_array = (unsigned int*)malloc(sizeof(unsigned int) * n);

		cudaMemcpy(h_array, d_array, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost);
		int i;

		//print the number of vectors
		fprintf(out, "%d\n", n);

		//print triangle indices
		for (i = 0; i<n; i++)
			fprintf(out, "%u ", h_array[i]);


		free(h_array);

		return true;

	}
	else return false;
}

template <typename T>
static bool writeArray(const T *d_array, std::string filename, int n, int nch=1) {

	if (d_array != NULL) {

		std::ofstream out(filename, std::ofstream::out);
		T *h_array = (T*)malloc(sizeof(T) * n * nch);

		cudaMemcpy(h_array, d_array, sizeof(T) * n * nch, cudaMemcpyDeviceToHost);
		int i;

		//print the number of vectors
		out << n <<std::endl;

		//print triangle indices
		for (i = 0; i < n; i++) {
			for (int j = 0; j < nch; j++)
				out << (T)h_array[i*nch + j]<<" ";
			out << std::endl;
		}
		free(h_array);

		return true;

	}
	else return false;
}
template <typename T>
static bool writeArray(const T *d_array, std::string filename, int w, int h, int nch ) {

	if (d_array != NULL) {
		w = w/2;
		int n = w*h; 

		std::ofstream out(filename, std::ofstream::out);
		T *h_array = (T*)malloc(sizeof(T) * n * nch);

		cudaMemcpy(h_array, d_array, sizeof(T) * n * nch, cudaMemcpyDeviceToHost);
		int i;

		//print the number of vectors
		out << n << std::endl;

		//print triangle indices
		for (i = 0; i < h; i++) {
			for (int k = 0; k < w; k++) {
				for (int j = 0; j < nch; j++)
					out << (T)h_array[(i*w + k)*nch + j] << " ";
			}
			out << std::endl;
		}
		free(h_array);

		return true;

	}
	else return false;
}

template <typename T>
static bool writeArrayHost(const T *h_array, std::string filename, int n, int nch = 1) {

	if (h_array != NULL) {

		std::ofstream out(filename, std::ofstream::out);
	
		int i;

		//print the number of vectors
		out << n << std::endl;

		//print triangle indices
		for (i = 0; i < n; i++) {
			for (int j = 0; j < nch; j++)
				out << h_array[i*nch + j] << " ";
			out << std::endl;
		}

		return true;

	}
	else return false;
}

static bool displayFloatMat(float *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	

	float *tmp = (float*)malloc(sizeof(float) * width * height);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32FC1, tmp);
	a = a * scale + offset;

	cv::imshow(windowname, a);
	//cv::waitKey(0);

	free(tmp);

	return true;

}

static bool displayFloatMatSingle(float *d_mat, std::string windowname, int width, int height, int nch, int selectedch, float scale = 1.0f, float offset = 0.0f) {

	

	float *tmp = (float*)malloc(sizeof(float) * width * height*nch);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height*nch, cudaMemcpyDeviceToHost);

	for (int i = 0; i < width*height; i++) 
		tmp[i] = tmp[i*nch + selectedch];

	cv::Mat a(height, width, CV_32FC1, tmp);

	cv::imshow(windowname, a * scale + offset);
	cv::waitKey(0);

	free(tmp);

	return true;

}

static bool displayDiffFloatMat(float *d_mat1, float *d_mat2, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	

	float *tmp1 = (float*)malloc(sizeof(float) * width * height);
	float *tmp2 = (float*)malloc(sizeof(float) * width * height);

	cudaMemcpy(tmp1, d_mat1, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	cudaMemcpy(tmp2, d_mat2, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	
	cv::Mat a1(height, width, CV_32FC1, tmp1);
	cv::Mat a2(height, width, CV_32FC1, tmp2);

	cv::imshow(windowname, (a1-a2) * scale + offset);
//	cv::waitKey(0);

	free(tmp1);
	free(tmp2);

	return true;

}

static bool writeDeltaFloat(float *delta_rot, float *delta_trans, std::string filename, int frame_cnt) {

	if (frame_cnt == 2) {
		std::ofstream out(filename, std::ofstream::out);

		out << delta_rot[0] << " " << delta_rot[1] << " "  << delta_rot[2] << " " << delta_trans[0] << " " << delta_trans[1] << " " << delta_trans[2] << std::endl;
	}
	else {
		std::ofstream out(filename, std::ofstream::out | std::ofstream::app);

		out << delta_rot[0] << " " << delta_rot[1] << " " << delta_rot[2] << " " << delta_trans[0] << " " << delta_trans[1] << " " << delta_trans[2] << std::endl;
	}
	return true;
}

static bool writeDiffFloatMat(float *d_mat1, float *d_mat2, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	

	float *tmp1 = (float*)malloc(sizeof(float) * width * height);
	float *tmp2 = (float*)malloc(sizeof(float) * width * height);

	cudaMemcpy(tmp1, d_mat1, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	cudaMemcpy(tmp2, d_mat2, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	cv::Mat a1(height, width, CV_32FC1, tmp1);
	cv::Mat a2(height, width, CV_32FC1, tmp2);

	cv::Mat b = (a1 - a2) * scale + offset;
	b *= 255;

	cv::imwrite(windowname, b);
	//	cv::waitKey(0);

	free(tmp1);
	free(tmp2);

	return true;

}
static bool displayNormalizedFloatMat(float *d_mat, std::string windowname, int width, int height) {

	float *tmp = (float*)malloc(sizeof(float) * width * height);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32FC1, tmp);

	double v_min, v_max;
	cv::minMaxLoc(a, &v_min, &v_max); 

	cv::imshow(windowname, (a-v_min)/(v_max-v_min));
	cv::waitKey(0);

	free(tmp);

	return true;

}

static bool displayFloat4Mat(float *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	float *tmp = (float*)malloc(sizeof(float) * width * height * 4);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height * 4, cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32FC4, tmp);

	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{

			((float*)a.data)[4 * (i*a.cols + j) + 3] = 1;

		}
	}

	cv::imshow(windowname, a * scale + offset);
//	cv::waitKey(0);

	free(tmp);

	return true;

}

static bool displayFloatMatHost(float *h_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	cv::Mat a(height, width, CV_32FC1, h_mat);

	cv::imshow(windowname, a * scale + offset);
	cv::waitKey(0);

	return true;

}


static bool displayFloat2Mat(float *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	float *tmp = (float*)malloc(sizeof(float) * width * height * 2);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height * 2, cudaMemcpyDeviceToHost);

  printf("%d %d\n", width, height);

	cv::Mat a(height, width, CV_32FC2, tmp);
	cv::Mat b(height, width, CV_32FC1);

  a = scale * a + offset;

	memset(b.data, 0, sizeof(float) * width *height);

	std::vector<cv::Mat> mats;
	
	cv::split(a, mats);

	mats.push_back(b);

	cv::Mat tmpmat;

	cv::merge(mats, tmpmat);

	cv::imshow(windowname, tmpmat);
//	cv::waitKey(0);

	free(tmp);

	return true;

}


static bool displayInt2Mat(int *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

  int *tmp = (int*)malloc(sizeof(int) * width * height * 2);
  cudaMemcpy(tmp, d_mat, sizeof(int) * width * height * 2, cudaMemcpyDeviceToHost);

  cv::Mat a(height, width, CV_32SC2, tmp);
  cv::Mat b(height, width, CV_32SC1);


  a.convertTo(a, CV_32FC2);
  b.convertTo(b, CV_32FC1);

  memset(b.data, 0, sizeof(int) * width * height);

  std::vector<cv::Mat> mats;

  cv::split(a, mats);

  mats.push_back(b);

  cv::Mat tmpmat;

  cv::merge(mats, tmpmat);


  cv::imshow(windowname, scale *  tmpmat + offset);
  cv::waitKey(0);

  free(tmp);

  return true;

}


static bool displayIntMat(int *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	int *tmp = (int*)malloc(sizeof(int) * width * height );
	cudaMemcpy(tmp, d_mat, sizeof(int) * width * height , cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32SC1, tmp);

	a.convertTo(a, CV_32FC1);
	
	cv::imshow(windowname, scale *  a + offset);
	cv::waitKey(0);

	free(tmp);

	return true;

}

static bool displayFloat3MatHost(float *h_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {
	

	cv::Mat a(height, width, CV_32FC3, h_mat);
	cv::imshow(windowname, scale *  a + offset);
	cv::waitKey(0);

	return true;

}


static bool displayFloat3Mat(float *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	float *tmp = (float*)malloc(sizeof(float) * width * height*3);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height*3, cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32FC3, tmp);

	cv::imshow(windowname, a * scale + offset);
	cv::waitKey(0);

	free(tmp);

	return true;

}

static bool displayFloatNMat(float *d_mat, std::string windowname, int width, int height, int nch, float scale = 1.0f, float offset = 0.0f) {

	float *tmp = (float*)malloc(sizeof(float) * width * height * nch);
	float *tmp_single = (float*)malloc(sizeof(float) * width * height);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height * nch, cudaMemcpyDeviceToHost);

	int i, j;

	for (int k = 0; k < nch; k++) {

		for (j = 0; j < height; j++) {
			for (i = 0; i < width; i++) {

				int index = j * width + i;

				tmp_single[index] = tmp[index*nch + k];
			}

		}
		cv::Mat b(height, width, CV_32FC1, tmp_single);
		cv::imshow(windowname, scale * b + offset);
		cv::waitKey(0);

	}

	free(tmp);
	free(tmp_single);

	return true;

}

static bool writeFloatMat(float *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	float *tmp = (float*)malloc(sizeof(float) * width * height);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32FC1, tmp);


	std::cout << windowname << std::endl;
	a = a * scale + offset;
	a *= 255;
	cv::imwrite(windowname, a);


	free(tmp);

	return true;

}

static bool writeNormalizedFloatMat(float *d_mat, std::string windowname, int width, int height) {

	float *tmp = (float*)malloc(sizeof(float) * width * height);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32FC1, tmp);

	double v_min, v_max;
	cv::minMaxLoc(a, &v_min, &v_max);

	cv::imwrite(windowname, 255 *(a - v_min) / (v_max - v_min));

	free(tmp);

	return true;

}

static bool writeFloat4Mat(float *d_mat, std::string windowname, int width, int height, float scale = 1.0f, float offset = 0.0f) {

	float *tmp = (float*)malloc(sizeof(float) * width * height * 4);

	cudaMemcpy(tmp, d_mat, sizeof(float) * width * height * 4, cudaMemcpyDeviceToHost);

	cv::Mat a(height, width, CV_32FC4, tmp);

	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{

			((float*)a.data)[4*(i*a.cols + j) + 3] = 1;

		}
	}
	a =  (a * scale + offset);

	cv::imwrite(windowname, a);

	free(tmp);

	return true;

}

#endif