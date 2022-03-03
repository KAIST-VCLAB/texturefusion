# TextureFusion: High-Quality Texture Acquisition for Real-Time RGB-D Scanning

#### [Project Page](http://vclab.kaist.ac.kr/cvpr2020p1/index.html) | [Paper](http://vclab.kaist.ac.kr/cvpr2020p1/TextureFusion-Paper.pdf) | [Presentation Video](https://www.youtube.com/watch?v=7LeecwSmx_A&feature=emb_title) | [Presentation](http://vclab.kaist.ac.kr/cvpr2020p1/TextureFusion-Slides.pdf)

Writers:   Joo Ho Lee (jhlee@vclab.kaist.ac.kr), Hyunho Ha (hhha@vclab.kaist.ac.kr), Min H. Kim (minhkim@kaist.ac.kr)

Institute: KAIST Visual Computing Laboratory

If you use our code for your academic work, please cite our paper:

@InProceedings{Lee_2020_CVPR,
author = {Joo Ho Lee and Hyunho Ha and Yue Dong and Xin Tong and Min H. Kim},
title = {TextureFusion: High-Quality Texture Acquisition for Real-Time RGB-D Scanning},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}

## Installation

Our implementation is based on the original voxel hashing (https://github.com/niessner/VoxelHashing).

To compile our codes, first obtain the entire source codes from the original [voxel hashing repository](https://github.com/niessner/VoxelHashing), including the Visual Studio project file. Then follow these steps:

1. In `VoxelHashing/DepthSensingCUDA/`, replace the folders `Include/` and `Source/` as well as the configuration files `zParameters*.txt` with the contents of our repository.

2. Replace `DepthSensing.cpp` and `DepthSensing.h` file with `texFusion.h`, `texFusion.cpp`, and `texFusion_main.cpp`

3. Configure the existing files in the `Source/*.h`, `Source/*.cpp`, and `Source/*.cu` to the Visual Studio project that does not exist in the voxel hashing repository.

Note that our source codes inherit the dependency of the original Voxel Hashing project.

Our work requires:
- [DirectX SDK June 2010](https://www.microsoft.com/en-us/download/details.aspx?id=6812)
- Both [Kinect SDK 1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40278) and [Kinect SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561)
- [CUDA](https://developer.nvidia.com/cuda-toolkit) (tested with version 10.1)
- Both [mLib](https://github.com/niessner/mLib) and mLibExternal (http://kaldir.vc.in.tum.de/mLib/mLibExternal.zip) with [OpenCV](https://opencv.org/) (tested with version 3.4.1): Note that the zip file, mLibExternal, includes other dependent libraries such as OpenNI 2 and Eigen.

Our code has been developed with Microsoft Visual Studio 2013 (VC++ 12) and Windows 10 (10.0.19041, build 19041) on a machine equipped with Intel i9-10920X (RAM: 64GB), NVIDIA TITAN RTX (RAM: 24GB). The main function is in `texFusion_main.cpp`.\

## Data

We provide the "fountain" dataset (originally created by Zhou and Koltun) compatible with our implementation
(link: http://vclab.kaist.ac.kr/cvpr2020p1/fountain_all.zip).

## Usage

Our program reads parameters from three files and you can change the program setting by changing them.

- zParametersDefault.txt

- zParametersTrackingDefault.txt

- zParametersWarpingDefault.txt

You can run our program with the provided fountain dataset.

Please set s_sensorIdx as 9 and s_binaryDumpSensorFile[0] as the fountain folder in zParametersDefault.txt.

## License

Joo Ho Lee, Hyunho Ha and Min H. Kim have developed this software and related documentation (the "Software"); confidential use in source form of the Software, without modification, is permitted provided that the following conditions are met:

Neither the name of the copyright holder nor the names of any contributors may be used to endorse or promote products derived from the Software without specific prior written permission.

The use of the software is for Non-Commercial Purposes only. As used in this Agreement, "Non-Commercial Purpose" means for the purpose of education or research in a non-commercial organisation only. "Non-Commercial Purpose" excludes, without limitation, any use of the Software for, as part of, or in any way in connection with a product (including software) or service which is sold, offered for sale, licensed, leased, published, loaned or rented. If you require a license for a use excluded by this agreement, please email [minhkim@kaist.ac.kr].

Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.

NB Our implementation is covered under the license of "Voxel Hashing" codes (CC BY-NC-SA 3.0). 

Please refer to license.txt for more details. 

## Contact

If you have any questions, please feel free to contact us.

Joo Ho Lee (jhlee@vclab.kaist.ac.kr)

Hyun Ho Ha (hhha@vclab.kaist.ac.kr)

Min H. Kim (minhkim@vclab.kaist.ac.kr)
