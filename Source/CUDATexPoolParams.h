#pragma once


#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"

__align__(16)

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"

struct TexPoolParams {
	TexPoolParams() {
	}

	unsigned int	m_texturePatchWidth;
	unsigned int	m_texturePatchSize;
	unsigned int	m_numTexturePatches;
	unsigned int	m_numTextureTileWidth;

	float m_minDepth;
	float m_maxDepth;

};