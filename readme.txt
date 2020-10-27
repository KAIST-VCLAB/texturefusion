/****************************************************************************

- Codename: TextureFusion: High-Quality Texture Acquisition for Real-Time RGB-D Scanning (CVPR 2020)

- Writers:   Joo Ho Lee(jhlee@vclab.kaist.ac.kr), Hyunho Ha (hhha@vclab.kaist.ac.kr), Min H. Kim (minhkim@kaist.ac.kr)

- Institute: KAIST Visual Computing Laboratory

- Bibtex:

@InProceedings{Lee_2020_CVPR,
author = {Joo Ho Lee and Hyunho Ha and Yue Dong and Xin Tong and Min H. Kim},
title = {TextureFusion: High-Quality Texture Acquisition for Real-Time RGB-D Scanning},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}

- Joo Ho Lee, Hyunho Ha and Min H. Kim have developed this software and related documentation
  (the "Software"); confidential use in source form of the Software,
  without modification, is permitted provided that the following
  conditions are met:
  1. Neither the name of the copyright holder nor the names of any
  contributors may be used to endorse or promote products derived from
  the Software without specific prior written permission.
  2. The use of the software is for Non-Commercial Purposes only. As
  used in this Agreement, "Non-Commercial Purpose" means for the
  purpose of education or research in a non-commercial organisation
  only. "Non-Commercial Purpose" excludes, without limitation, any use
  of the Software for, as part of, or in any way in connection with a
  product (including software) or service which is sold, offered for
  sale, licensed, leased, published, loaned or rented. If you require
  a license for a use excluded by this agreement,
  please email [minhkim@kaist.ac.kr].

- License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License

- Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE
  SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT
  LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
  PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY
  DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
  THIS SOFTWARE OR ITS DERIVATIVES

*****************************************************************************/

--------------------------------------------------------------------------------------------------
Information
--------------------------------------------------------------------------------------------------

Our implementation is based on "Voxel Hashing" repository ( https://github.com/niessner/VoxelHashing ).
So, our implementation is covered under the license of "Voxel Hashing" codes.

--------------------------------------------------------------------------------------------------
Background
--------------------------------------------------------------------------------------------------

For the details, please see our paper.

Title: TextureFusion: High-Quality Texture Acquisition for Real-Time RGB-D Scanning
Author: Joo Ho Lee, Ha Hyunho, Yue Dong, Xin Ton, Min H. Kim
Conference: CVPR 2020

Please cite this paper if you use this code in academic publication.
s
@InProceedings{Lee_2020_CVPR,
author = {Joo Ho Lee and Hyunho Ha and Yue Dong and Xin Tong and Min H. Kim},
title = {TextureFusion: High-Quality Texture Acquisition for Real-Time RGB-D Scanning},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}


--------------------------------------------------------------------------------------------------
Contents
--------------------------------------------------------------------------------------------------

Newly added codes
-CUDABuildLinearSystemRGBLocal.cpp
-CUDABuildLinearSystemRGBLocal.cu
-CUDABuildLinearSystemRGBLocal.h
-CUDACameraTrackingLocalRGB.cpp
-CUDACameraTrackingLocalRGB.cu
-CUDACameraTrackingLocalRGB.h
-CUDATexUpdate.cu
-CUDATexUpdate.cpp
-CUDATexUpdate.h
-CUDATexUpdateParams.h
-texUpdateUtil.h
-texturePool.h
-modelDefine.h
-sensordatazhou.cpp
-sensordatazhou.h
-SensorDataReaderZhou.cpp
-SensorDataReaderZhou.h
-CudaImageUtil.cu
-CudaImageUtil.h
-cudaDebug.h
-texFusion.cpp
-texFusion.h
-texFusion_main.cpp
-GlobalWarpingState.cpp
-GlobalWarpingState.h

Modified codes
-CUDARayCastSDF.cpp
-CUDARayCastSDF.cu
-CUDARayCastSDF.h
-CUDARayCastParams.h
-CUDASceneRepHashSDF.cpp
-CUDASceneRepHashSDF.cu
-CUDASceneRepHashSDF.h
-RayCastSDFUtil.h
-GlobalAppState.h


--------------------------------------------------------------------------------------------------
How to run
--------------------------------------------------------------------------------------------------

You have to download and run voxel hashing code. Then, you should exclude depthsensing.cpp and depthsensing.h,
 and compile with the main function in texFusion_main.cpp.

--------------------------------------------------------------------------------------------------
Contact
--------------------------------------------------------------------------------------------------

Joo Ho Lee (jhlee@vclab.kaist.ac.kr)
Hyun Ho Ha (hhha@vclab.kaist.ac.kr)
Min H. Kim (minhkim@vclab.kaist.ac.kr)
