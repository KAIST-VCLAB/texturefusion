#pragma once

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

__align__(16)
struct TexUpdateParams {

	//blending
	float m_angleThreshold;
	float m_angleThreshold_update;
	float m_angleThreshold_depth;
	float m_integrationWeightSample;
	float m_integrationWeightMax;
	
	//erode
	int m_erode_iter_occdepth;
	int m_erode_iter_stretch_box;
	int m_erode_iter_stretch_gauss;
	
	int m_screen_boundary_width;
	
	float m_sigma_depth; 
	float m_sigma_angle;
	float m_sigma_area;

	int m_erode_iter_gauss_warping;
	int m_erode_iter_box_warping;
	float m_erode_sigma_stretch;
	
	int m_warp_mode;

	int m_width, m_height;

};
