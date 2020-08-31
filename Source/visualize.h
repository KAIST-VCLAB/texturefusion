#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"


void displayMotion(float *d_image, std::string windowname, float2 *d_motion, int image_w, int image_h, int cell_w, int node_w, int node_h) {

	float *h_image;
	float2 *h_motion;

	printf("%d %d %d %d %d\n", image_w, image_h, cell_w, node_w, node_h);

	h_image = (float*) malloc(sizeof(float) * image_w * image_h);
	h_motion = (float2*) malloc(sizeof(float2) * node_w * node_h);

	cudaDeviceSynchronize();
	cudaMemcpy(h_image, d_image, sizeof(float) *image_w *image_h, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_motion, d_motion, sizeof(float2) * node_w * node_h, cudaMemcpyDeviceToHost);


	cv::Mat image_mat(image_h, image_w, CV_32FC1, h_image);

	//void arrowedLine(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness = 1, int line_type = 8, int shift = 0, double tipLength = 0.1)


	for (int i = 0; i < node_h; i++) {
		for (int j = 0; j < node_w; j++) {

			int index = i * node_w + j;
			
//			if (j*cell_w < image_w && i*cell_w < image_h) {
				cv::Point start(j*cell_w, i*cell_w);
				cv::Point motion(h_motion[index].x, h_motion[index].y);
				cv::Point end;
				end = start + motion;
			//	printf("%f %f)", motion.x, motion.y);

				cv::arrowedLine(image_mat, start, end, cv::Scalar(0.));
//			}
		}
	}

	cv::imshow(windowname, image_mat);
	cv::waitKey(0);

	free(h_image);
	free(h_motion);

}


void displayMotionWithDelta(float *d_image, std::string windowname, float2 *d_motion, float2 *d_delta, int image_w, int image_h, int cell_w, int node_w, int node_h) {

	float *h_image;
	float2 *h_motion;
	float2 *h_delta;

	printf("%d %d %d %d %d\n", image_w, image_h, cell_w, node_w, node_h);

	h_image = (float*)malloc(sizeof(float) * image_w * image_h);
	h_motion = (float2*)malloc(sizeof(float2) * node_w * node_h);
	h_delta = (float2*)malloc(sizeof(float2) * node_w * node_h);

	cudaDeviceSynchronize();
	cudaMemcpy(h_image, d_image, sizeof(float) *image_w *image_h, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_motion, d_motion, sizeof(float2) * node_w * node_h, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_delta, d_delta, sizeof(float2) * node_w * node_h, cudaMemcpyDeviceToHost);


	cv::Mat image_mat(image_h, image_w, CV_32FC1, h_image);

	//void arrowedLine(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness = 1, int line_type = 8, int shift = 0, double tipLength = 0.1)


	for (int i = 0; i < node_h; i++) {
		for (int j = 0; j < node_w; j++) {

			int index = i * node_w + j;

			//			if (j*cell_w < image_w && i*cell_w < image_h) {
			cv::Point start(j*cell_w, i*cell_w);
			cv::Point motion(h_motion[index].x + h_delta[index].x, h_motion[index].y + h_delta[index].y);

			//motion *= 100;
			//printf("%f %f), ", h_delta[index].x, h_delta[index].y);
			cv::Point end;
			end = start + motion;

			cv::arrowedLine(image_mat, start, end, cv::Scalar(0.));
			//			}
		}
	}

	cv::imshow(windowname, image_mat);
	cv::waitKey(0);

	free(h_image);
	free(h_motion);

}

void writeMotion(float *d_image, std::string windowname, float2 *d_motion, int image_w, int image_h,int tar_w, int tar_h, int cell_w, int node_w, int node_h) {

	float *h_image;
	float2 *h_motion;

	printf("%d %d %d %d %d\n", image_w, image_h, cell_w, node_w, node_h);

	h_image = (float*)malloc(sizeof(float) * image_w * image_h);
	h_motion = (float2*)malloc(sizeof(float2) * node_w * node_h);

	cudaDeviceSynchronize();
	cudaMemcpy(h_image, d_image, sizeof(float) *image_w *image_h, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_motion, d_motion, sizeof(float2) * node_w * node_h, cudaMemcpyDeviceToHost);


	cv::Mat image_mat(image_h, image_w, CV_32FC1, h_image);
//	cv::Mat image_mat(image_h, image_w, CV_32FC1,cv::Scalar(0));

	image_mat *= 0.5;

	int top, bottom, left, right;
	int borderType = cv::BORDER_CONSTANT;

	cv::resize(image_mat, image_mat, cv::Size(tar_w, tar_h), CV_INTER_AREA);



	float ratiow = (float)tar_w / (float)image_w;
	float ratioh = (float)tar_h / (float)image_h;

	top = (int)(0.05*(image_mat.rows)); bottom = (int)(0.05*image_mat.rows) + ((node_h-1) * cell_w * ratioh - tar_h) ;
	left = (int)(0.05*image_mat.cols); right = (int)(0.05*image_mat.cols) + ((node_w-1) * cell_w * ratiow - tar_w);

	//void arrowedLine(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness = 1, int line_type = 8, int shift = 0, double tipLength = 0.1)
	cv::Scalar value = cv::Scalar(0.);
	cv::copyMakeBorder(image_mat, image_mat, top, bottom, left, right, borderType, value);


	for (int i = 0; i < node_h; i++) {
		for (int j = 0; j < node_w; j++) {

			int index = i * node_w + j;

			//			if (j*cell_w < image_w && i*cell_w < image_h) {
			cv::Point2f start(j*cell_w, i*cell_w);
			cv::Point2f motion(h_motion[index].x, h_motion[index].y);
			cv::Point2f end;
			end = start + motion;

			start.x *= ratiow;
			start.y *= ratioh;
			end.x *= ratiow;
			end.y *= ratioh;

			start += cv::Point2f(left, top);
			end += cv::Point2f(left, top);
			//printf("%f %f)", motion.x, motion.y);

			cv::arrowedLine(image_mat, start, end, cv::Scalar(1.));
			//			}
		}
	}


	image_mat *= 255;
	cv::imwrite(windowname, image_mat);

	free(h_image);
	free(h_motion);

}


void writeMotionWithDelta(float *d_image, std::string windowname, float2 *d_motion, float2 *d_delta, int image_w, int image_h, int cell_w, int node_w, int node_h) {

	float *h_image;
	float2 *h_motion;
	float2 *h_delta;

	printf("%d %d %d %d %d\n", image_w, image_h, cell_w, node_w, node_h);

	h_image = (float*)malloc(sizeof(float) * image_w * image_h);
	h_motion = (float2*)malloc(sizeof(float2) * node_w * node_h);
	h_delta = (float2*)malloc(sizeof(float2) * node_w * node_h);

	cudaDeviceSynchronize();
	cudaMemcpy(h_image, d_image, sizeof(float) *image_w *image_h, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_motion, d_motion, sizeof(float2) * node_w * node_h, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_delta, d_delta, sizeof(float2) * node_w * node_h, cudaMemcpyDeviceToHost);


	cv::Mat image_mat(image_h, image_w, CV_32FC1, h_image);

	//void arrowedLine(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness = 1, int line_type = 8, int shift = 0, double tipLength = 0.1)


	for (int i = 0; i < node_h; i++) {
		for (int j = 0; j < node_w; j++) {

			int index = i * node_w + j;

			//			if (j*cell_w < image_w && i*cell_w < image_h) {
			cv::Point start(j*cell_w, i*cell_w);
			cv::Point motion(h_motion[index].x + h_delta[index].x, h_motion[index].y + h_delta[index].y);

			//motion *= 100;
			//printf("%f %f), ", h_delta[index].x, h_delta[index].y);
			cv::Point end;
			end = start + motion;

			cv::arrowedLine(image_mat, start, end, cv::Scalar(0.));
			//			}
		}
	}

	image_mat *= 255;
	cv::imwrite(windowname, image_mat);

	free(h_image);
	free(h_motion);

}
void writeFlow() {

}

//void writeFlow