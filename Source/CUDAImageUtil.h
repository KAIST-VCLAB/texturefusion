#pragma once

extern "C" void gaussFilterAlphaMap(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height);
extern "C" void gaussBlur(float* d_output, float* d_input, float *d_mask, float sigmaD, unsigned int width, unsigned int height);
extern "C" void gaussBlurWOMask(float* d_output, float* d_input, float sigmaD, unsigned int width, unsigned int height);
extern "C" void downSample(float* d_output, float* d_input, unsigned int width, unsigned int height);
extern "C" void downSampleMask(float* d_output, float* d_input, unsigned int width, unsigned int height);
extern "C" void computeSobelGradient(float2 * d_output, float *d_input, unsigned int width, unsigned int height);
extern "C" void computeGradient3(float2 * d_output, float *d_input, unsigned int width, unsigned int height);
extern "C" void computeGradient2(float2 * d_output, float *d_input, unsigned int width, unsigned int height);

extern "C" void convertColor2Gray(float* d_output, float4* d_input, unsigned int width, unsigned int height);
extern "C" void opticalFlowToColor(float2 *d_flow, float4* d_flow_colorization, unsigned int width, unsigned int height);
