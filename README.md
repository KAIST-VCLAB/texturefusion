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

Our implementation is based on the voxel hashing repository (https://github.com/niessner/VoxelHashing).
Please follow the instruction to compile the voxel hashing repository.
After compiling the voxel hashing repository, replace the 'Source' folder with the 'Source' folder of our implementation.
Then, add all codes in the project and run our code. 
The main function is in 'texture_main.cpp'.

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

Our implementation is covered under the license of "Voxel Hashing" codes (CC BY-NC-SA 3.0). Please refer to license.txt for more details. 

## Contact

If you have any questions, please feel free to contact us.

Joo Ho Lee (jhlee@vclab.kaist.ac.kr)

Hyun Ho Ha (hhha@vclab.kaist.ac.kr)

Min H. Kim (minhkim@vclab.kaist.ac.kr)