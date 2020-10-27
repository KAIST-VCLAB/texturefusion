#include "stdafx.h"

#include "VoxelUtilHashSDF.h"
#include "RayCastSDFUtil.h"
#include "CUDAMarchingCubesHashSDF.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\core\cuda.hpp>
#include <opencv2\cudaoptflow.hpp>
#include <opencv2\cudaarithm.hpp>

extern "C" void resetMarchingCubesCUDA(MarchingCubesData& data);
extern "C" void extractIsoSurfaceCUDA(const HashData& hashData,
	const RayCastData& rayCastData,
	const MarchingCubesParams& params,
	MarchingCubesData& data,
	const TexPoolData& texPoolData);


extern "C" void extractIsoSurfacePass1CUDA(const HashData& hashData, const RayCastData& rayCastData, const MarchingCubesParams& params, MarchingCubesData& data);
extern "C" void extractIsoSurfacePass2CUDA(const HashData& hashData, const RayCastData& rayCastData, const MarchingCubesParams& params, MarchingCubesData& data, const TexPoolData& texPoolData, unsigned int numOccupiedBlocks);
extern "C" void exportTexture(uchar *d_output, uint *d_texelAddress, Texel *d_input, unsigned int tex_width, unsigned int tex_height, unsigned int patch_width, unsigned int patch_size, unsigned int num_patches, unsigned int max_num_patches);

void CUDAMarchingCubesHashSDF::create(const MarchingCubesParams& params)
{
	m_params = params;
	m_data.allocate(m_params);

	resetMarchingCubesCUDA(m_data);
}

void CUDAMarchingCubesHashSDF::destroy(void)
{
	m_data.free();
}

void CUDAMarchingCubesHashSDF::copyTrianglesToCPU(TexPoolData texPoolData, TexPoolParams texPoolParams) {

	//could  be a bit more efficient here; rather than allocating so much memory; just allocate depending on the triangle size;
	MarchingCubesData cpuData = m_data.copyToCPU();
	std::cout << "Copy GPU To CPU" << std::endl;
	unsigned int nTriangles = *cpuData.d_numTriangles;

	if (nTriangles >= m_params.m_maxNumTriangles) throw MLIB_EXCEPTION("not enough memory to store triangles for chunk; increase s_marchingCubesMaxNumTriangles");

	std::cout << "Marching Cubes: #triangles = " << nTriangles << std::endl;

	uint m_numTexturePatch = texPoolParams.m_numTexturePatches;
	uint h_numTextureTile = 0;
	
	std::vector<uint> textureIndex, uniqueTextureIndex, uniqueTextureIndexInverse;
	if (!GlobalAppState::get().s_offlineProcessing) {
		throw MLIB_EXCEPTION("Current version does not support streaming out processing.");

		if (nTriangles != 0) {
			textureIndex.resize(3 * nTriangles);
			unsigned int baseIdx = (unsigned int)m_meshData.m_Vertices.size();
			m_meshData.m_Vertices.resize(baseIdx + 3 * nTriangles);
			m_meshData.m_Colors.resize(baseIdx + 3 * nTriangles);
			// For the texture
			m_meshData.m_TextureCoords.resize(baseIdx + 3 * nTriangles);

			vec3f* vc = (vec3f*)cpuData.d_triangles;

			for (unsigned int i = 0; i < 3 * nTriangles; i++) {

				m_meshData.m_Vertices[baseIdx + i] = vc[3 * i + 0];
				m_meshData.m_Colors[baseIdx + i] = vec4f(vc[3 * i + 1]);
				m_meshData.m_TextureCoords[baseIdx + i] = vec2f(vc[3 * i + 2]);

				uint texind = uint(vc[3 * i + 2].z + 0.001);
				textureIndex[i] = texind;
			}

			uint textureTileWidth = texPoolParams.m_numTextureTileWidth;
			while (h_numTextureTile >= textureTileWidth * textureTileWidth) {
				textureTileWidth *= 2;
				printf("Resize the texture tile width: %d\n", textureTileWidth);
			}

			uniqueTextureIndexInverse.resize(texPoolParams.m_numTexturePatches);
			memset(uniqueTextureIndexInverse.data(), texPoolParams.m_numTexturePatches + 1, sizeof(uint) * texPoolParams.m_numTexturePatches);

			for (unsigned int i = 0; i < uniqueTextureIndex.size(); i++)
				uniqueTextureIndexInverse[uniqueTextureIndex[i]] = i;

			//assign texture coordinates to each vertex.
			for (unsigned int i = 0; i < 3 * nTriangles; i++) {
				unsigned int index = 0;
				index = uniqueTextureIndexInverse[textureIndex[i]];

				if (index > texPoolParams.m_numTexturePatches)
					throw MLIB_EXCEPTION("not enough memory to store texture for heap; increase s_texPoolNumPatches");

				unsigned int patch_y = unsigned int(index / textureTileWidth);
				unsigned int patch_x = unsigned int(index % textureTileWidth);
				m_meshData.m_TextureCoords[i] = (m_meshData.m_TextureCoords[i] + vec2f(patch_x, patch_y) * (texPoolParams.m_texturePatchWidth + 2)) / textureTileWidth / (texPoolParams.m_texturePatchWidth + 2);
			}
			printf("Texcoord indexing finish\n");

			uchar *h_textureImg = (uchar *)malloc(sizeof(uchar) * 3 * (texPoolParams.m_texturePatchWidth + 2) * (texPoolParams.m_texturePatchWidth + 2) * textureTileWidth * textureTileWidth);

			uchar *d_textureImg;
			unsigned int *d_textureAddress;
			cutilSafeCall(cudaMalloc(&d_textureImg, sizeof(uchar) * 3 * (texPoolParams.m_texturePatchWidth + 2) * (texPoolParams.m_texturePatchWidth + 2) * textureTileWidth * textureTileWidth));
			cudaMemset(d_textureImg, 0, sizeof(uchar) * 3 * (texPoolParams.m_texturePatchWidth + 2) * (texPoolParams.m_texturePatchWidth + 2) * textureTileWidth * textureTileWidth);
			cutilSafeCall(cudaMalloc(&d_textureAddress, sizeof(unsigned int) * textureTileWidth * textureTileWidth));
			cudaMemset(d_textureAddress, 0, sizeof(unsigned int) * textureTileWidth * textureTileWidth);

			cudaMemcpy(d_textureAddress, uniqueTextureIndex.data(), sizeof(unsigned int) * h_numTextureTile, cudaMemcpyHostToDevice);
			exportTexture(d_textureImg, d_textureAddress, texPoolData.d_texPatches, textureTileWidth, textureTileWidth, texPoolParams.m_texturePatchWidth, texPoolParams.m_texturePatchSize, h_numTextureTile, texPoolParams.m_numTexturePatches);
			cudaMemcpy(h_textureImg, d_textureImg, sizeof(uchar) * 3 * (texPoolParams.m_texturePatchWidth + 2) * (texPoolParams.m_texturePatchWidth + 2) * textureTileWidth * textureTileWidth, cudaMemcpyDeviceToHost);

			free(h_textureImg);
			cutilSafeCall(cudaFree(d_textureImg));
			cutilSafeCall(cudaFree(d_textureAddress));
		}
	}
	else {
		//some sequences exhaust cpu memory... -> merge first
		std::cout << "Generate texture with textile width: " << texPoolParams.m_numTextureTileWidth << ", Patch width: " << texPoolParams.m_texturePatchWidth << std::endl;
		if (nTriangles != 0) {
			MeshDataf md;

			md.m_Vertices.resize(3 * nTriangles);
			md.m_Colors.resize(3 * nTriangles);
			
			// For the texture
			md.m_TextureCoords.resize(3 * nTriangles);
			textureIndex.resize(3 * nTriangles);

			vec3f* vc = (vec3f*)cpuData.d_triangles;
			for (unsigned int i = 0; i < 3 * nTriangles; i++) {
				md.m_Vertices[i] = vc[3 * i + 0];
				md.m_Colors[i] = vec4f(vc[3 * i + 1]);

				md.m_TextureCoords[i] = vec2f(vc[3 * i + 2]) + vec2f(1.0f);
				if (md.m_TextureCoords[i].x <= 1.0f) md.m_TextureCoords[i].x += 0.001f;
				if (md.m_TextureCoords[i].y <= 1.0f) md.m_TextureCoords[i].y += 0.001f;
				if (md.m_TextureCoords[i].x >= texPoolParams.m_texturePatchWidth) md.m_TextureCoords[i].x -= 0.001f;
				if (md.m_TextureCoords[i].y >= texPoolParams.m_texturePatchWidth) md.m_TextureCoords[i].y -= 0.001f;
				textureIndex[i] = uint(vc[3 * i + 2].z);
			}

			//remove empty index space in the texture.
			uniqueTextureIndex = textureIndex;
			std::sort(uniqueTextureIndex.begin(), uniqueTextureIndex.end());
			auto last = std::unique(uniqueTextureIndex.begin(), uniqueTextureIndex.end());
			uniqueTextureIndex.erase(last, uniqueTextureIndex.end());

			h_numTextureTile = uniqueTextureIndex.size();
			printf("Number of unique texture index: %d\n", h_numTextureTile);

			uint textureTileWidth = texPoolParams.m_numTextureTileWidth;
			while (h_numTextureTile >= textureTileWidth * textureTileWidth) {
				textureTileWidth *= 2;
				printf("Resize the texture tile width: %d\n", textureTileWidth);
			}

			uniqueTextureIndexInverse.resize(texPoolParams.m_numTexturePatches);
			memset(uniqueTextureIndexInverse.data(), texPoolParams.m_numTexturePatches + 1, sizeof(uint) * texPoolParams.m_numTexturePatches);

			for (unsigned int i = 0; i < uniqueTextureIndex.size(); i++)
				uniqueTextureIndexInverse[uniqueTextureIndex[i]] = i;

			//assign texture coordinates to each vertex.
			for (unsigned int i = 0; i < 3 * nTriangles; i++) {
				unsigned int index = 0;
				index = uniqueTextureIndexInverse[textureIndex[i]];
				
				if (index > texPoolParams.m_numTexturePatches)
					throw MLIB_EXCEPTION("not enough memory to store texture for heap; increase s_texPoolNumPatches");

				unsigned int patch_y = unsigned int(index / textureTileWidth);
				unsigned int patch_x = unsigned int(index % textureTileWidth);
				md.m_TextureCoords[i] = (md.m_TextureCoords[i] + vec2f(patch_x, patch_y) * (float)(texPoolParams.m_texturePatchWidth + 2)) / (float)textureTileWidth / (float)(texPoolParams.m_texturePatchWidth + 2.0f);
			}
			printf("Texcoord indexing finish\n");

			//create index buffer (required for merging the triangle soup)
			md.m_FaceIndicesVertices.resize(md.m_Vertices.size());
			md.m_FaceIndicesTextureCoords.resize(md.m_Vertices.size());
			for (unsigned int i = 0; i < (unsigned int)md.m_Vertices.size() / 3; i++) {
				md.m_FaceIndicesVertices[i][0] = 3 * i + 0;
				md.m_FaceIndicesVertices[i][1] = 3 * i + 1;
				md.m_FaceIndicesVertices[i][2] = 3 * i + 2;
			}
			printf("Face index finish\n");

			
			uchar *h_textureImg = (uchar *)malloc(sizeof(uchar) * 3 * (texPoolParams.m_texturePatchWidth + 2) * (texPoolParams.m_texturePatchWidth + 2) * textureTileWidth * textureTileWidth);
			
			uchar *d_textureImg;
			unsigned int *d_textureAddress;
			cutilSafeCall(cudaMalloc(&d_textureImg, sizeof(uchar) * 3 * (texPoolParams.m_texturePatchWidth + 2) * (texPoolParams.m_texturePatchWidth + 2) * textureTileWidth * textureTileWidth));
			cudaMemset(d_textureImg, 0, sizeof(uchar) * 3 * (texPoolParams.m_texturePatchWidth + 2) * (texPoolParams.m_texturePatchWidth + 2) * textureTileWidth * textureTileWidth);
			cutilSafeCall(cudaMalloc(&d_textureAddress, sizeof(unsigned int) * textureTileWidth * textureTileWidth));
			cudaMemset(d_textureAddress, 0, sizeof(unsigned int) * textureTileWidth * textureTileWidth);

			cudaMemcpy(d_textureAddress, uniqueTextureIndex.data(), sizeof(unsigned int) * h_numTextureTile, cudaMemcpyHostToDevice);
			exportTexture(d_textureImg, d_textureAddress, texPoolData.d_texPatches, textureTileWidth, textureTileWidth, texPoolParams.m_texturePatchWidth, texPoolParams.m_texturePatchSize, h_numTextureTile, texPoolParams.m_numTexturePatches);
			cudaMemcpy(h_textureImg, d_textureImg, sizeof(uchar) * 3 * (texPoolParams.m_texturePatchWidth + 2) * (texPoolParams.m_texturePatchWidth + 2) * textureTileWidth * textureTileWidth, cudaMemcpyDeviceToHost);

			printf("Generate texture\n");

			unsigned int size = textureTileWidth * textureTileWidth * (texPoolParams.m_texturePatchWidth + 2) * (texPoolParams.m_texturePatchWidth + 2);

			std::string folderName = "Scans/";
			std::string textureFileName = GlobalAppState::get().s_sceneName + GlobalAppState::get().export_surfix[GlobalAppState::get().s_optimizationIdx] + "_texture.png";
			std::string mtlFileName = GlobalAppState::get().s_sceneName + GlobalAppState::get().export_surfix[GlobalAppState::get().s_optimizationIdx] + ".mtl";

			std::string folderTextureFileName = folderName + GlobalAppState::get().s_sceneName + GlobalAppState::get().export_surfix[GlobalAppState::get().s_optimizationIdx] + "_texture.png";
			std::string folderMtlFileName = folderName + GlobalAppState::get().s_sceneName + GlobalAppState::get().export_surfix[GlobalAppState::get().s_optimizationIdx] + ".mtl";

			while (util::fileExists(folderTextureFileName)) {
				std::string path = util::directoryFromPath(folderTextureFileName);
				std::string curr = util::fileNameFromPath(folderTextureFileName);
				std::string ext = util::getFileExtension(curr);
				curr = util::removeExtensions(curr);
				std::string base = util::getBaseBeforeNumericSuffix(curr);
				unsigned int num = util::getNumericSuffix(curr);
				if (num == (unsigned int)-1) {
					num = 0;
				}
				folderTextureFileName = path + base + std::to_string(num + 1) + ".png" ;
				textureFileName = base + std::to_string(num + 1) + ".png";
			}

			while (util::fileExists(folderMtlFileName)) {
				std::string path = util::directoryFromPath(folderMtlFileName);
				std::string curr = util::fileNameFromPath(folderMtlFileName);
				std::string ext = util::getFileExtension(curr);
				curr = util::removeExtensions(curr);
				std::string base = util::getBaseBeforeNumericSuffix(curr);
				unsigned int num = util::getNumericSuffix(curr);
				if (num == (unsigned int)-1) {
					num = 0;
				}
				folderMtlFileName = path + base + std::to_string(num + 1) + ".mtl";
				mtlFileName = base + std::to_string(num + 1) + ".mtl";
			}

			GlobalAppState::get().export_mtlfilename.push_back(mtlFileName);

			std::ofstream mtlout(folderMtlFileName, std::ios::out);
			mtlout << "newmtl MAT_F436B0\n" 
				<< "\tKa 1.0 1.0 1.0\n"
				<< "\tKd 1.0 1.0 1.0\n"
				<< "\tKs 1.0 1.0 1.0\n"
				<< "\tillum 1.0\n"
				<< "\tNs 1\n"
				<< "\tmap_Kd " << textureFileName;
			mtlout.close();

			cv::Mat textureMat(textureTileWidth * (texPoolParams.m_texturePatchWidth + 2), textureTileWidth * (texPoolParams.m_texturePatchWidth + 2), CV_8UC3, h_textureImg);
			cv::cvtColor(textureMat, textureMat, cv::COLOR_RGB2BGR);
			cv::imwrite(folderTextureFileName, textureMat);

			printf("Export texture\n");
			//md.mergeCloseVertices(0.001f, true);
			//md.removeDuplicateFaces();

			md.m_FaceIndicesTextureCoords = md.m_FaceIndicesVertices;
			if (md.m_FaceIndicesVertices.size() > 0)
				m_meshData.merge(md);

			md.m_TextureCoords.clear();
			md.m_Colors.clear();
			md.m_FaceIndicesTextureCoords.clear();
			if (md.m_FaceIndicesVertices.size() > 0)
				m_meshOnlyData.merge(md);

			cutilSafeCall(cudaFree(d_textureImg));
			cutilSafeCall(cudaFree(d_textureAddress));
			free(h_textureImg);
		}
	}

	cpuData.free();
}


void CUDAMarchingCubesHashSDF::saveMesh(const std::string& filename, const mat4f *transform /*= NULL*/, bool overwriteExistingFile /*= false*/)
{
	std::string folder = util::directoryFromPath(filename);
	if (!util::directoryExists(folder)) {
		util::makeDirectory(folder);
	}

	//std::string actualFilename = filename;
	std::string actualFilename = "Scans/" + GlobalAppState::get().s_sceneName + GlobalAppState::get().export_surfix[GlobalAppState::get().s_optimizationIdx] + ".obj";
	std::string actualFilenamePLY = "Scans/" + GlobalAppState::get().s_sceneName + GlobalAppState::get().export_surfix[GlobalAppState::get().s_optimizationIdx] + ".ply";
	if (!overwriteExistingFile) {
		while (util::fileExists(actualFilename)) {
			std::string path = util::directoryFromPath(actualFilename);
			std::string curr = util::fileNameFromPath(actualFilename);
			std::string ext = util::getFileExtension(curr);
			curr = util::removeExtensions(curr);
			std::string base = util::getBaseBeforeNumericSuffix(curr);
			unsigned int num = util::getNumericSuffix(curr);
			if (num == (unsigned int)-1) {
				num = 0;
			}
			actualFilename = path + base + std::to_string(num + 1) + "." + ext;
			actualFilenamePLY = path + base + std::to_string(num + 1) + ".ply";
		}
	}

	//create index buffer (required for merging the triangle soup)
	if (!m_meshData.hasVertexIndices()) {
		m_meshData.m_FaceIndicesVertices.resize(m_meshData.m_Vertices.size());
		m_meshData.m_FaceIndicesTextureCoords.resize(m_meshData.m_Vertices.size());
		for (unsigned int i = 0; i < (unsigned int)m_meshData.m_Vertices.size() / 3; i++) {
			m_meshData.m_FaceIndicesVertices[i][0] = 3 * i + 0;
			m_meshData.m_FaceIndicesVertices[i][1] = 3 * i + 1;
			m_meshData.m_FaceIndicesVertices[i][2] = 3 * i + 2;
			m_meshData.m_FaceIndicesTextureCoords[i][0] = 3 * i + 0;
			m_meshData.m_FaceIndicesTextureCoords[i][1] = 3 * i + 1;
			m_meshData.m_FaceIndicesTextureCoords[i][2] = 3 * i + 2;
		}
	}
	std::cout << "size before:\t" << m_meshData.m_Vertices.size() << std::endl;

	m_meshData.removeDuplicateVertices();
	//m_meshData.mergeCloseVertices(0.00001f);
	std::cout << "merging close vertices... ";
	//m_meshData.mergeCloseVertices(0.0001f, true);
	std::cout << "done!" << std::endl;
	std::cout << "removing duplicate faces... ";
	//m_meshData.removeDuplicateFaces();
	std::cout << "done!" << std::endl;

	std::cout << "size after:\t" << m_meshData.m_Vertices.size() << std::endl;

	if (transform) {
		m_meshData.applyTransform(mat4f::identity()); // *transform);
	}
	std::string mtlfilename = GlobalAppState::get().export_mtlfilename.back();
	GlobalAppState::get().export_mtlfilename.pop_back();

	std::cout << "saving mesh (" << actualFilename << ") ...";
	//MeshIOf::saveToFile(actualFilename, m_meshData, mtlfilename);
	std::cout << "done!" << std::endl;

	m_meshOnlyData.removeDuplicateVertices();
	std::cout << "merging close vertices... ";
	//m_meshOnlyData.mergeCloseVertices(0.0001f, true);
	std::cout << "done!" << std::endl;
	std::cout << "removing duplicate faces... ";
	//m_meshOnlyData.removeDuplicateFaces();
	std::cout << "done!" << std::endl;

	std::cout << "size after:\t" << m_meshOnlyData.m_Vertices.size() << std::endl;

	if (transform) {
		m_meshOnlyData.applyTransform(mat4f::identity()); // *transform);
	}

	std::cout << "saving mesh (" << actualFilenamePLY << ") ...";
	MeshIOf::saveToFile(actualFilenamePLY, m_meshOnlyData);
	std::cout << "done!" << std::endl;


	clearMeshBuffer();

}


void CUDAMarchingCubesHashSDF::extractIsoSurface(CUDASceneRepChunkGrid& chunkGrid, const RayCastData& rayCastData, const TexPoolData& texPoolData, const TexPoolParams& texPoolParams, const vec3f& camPos, float radius)
{

	chunkGrid.stopMultiThreading();

	const vec3i& minGridPos = chunkGrid.getMinGridPos();
	const vec3i& maxGridPos = chunkGrid.getMaxGridPos();

	clearMeshBuffer();

	chunkGrid.streamOutToCPUAll();

	for (int x = minGridPos.x; x < maxGridPos.x; x++) {
		for (int y = minGridPos.y; y < maxGridPos.y; y++) {
			for (int z = minGridPos.z; z < maxGridPos.z; z++) {

				vec3i chunk(x, y, z);

				if (chunkGrid.containsSDFBlocksChunk(chunk)) {

					std::cout << "Marching Cubes on chunk (" << x << ", " << y << ", " << z << ") " << std::endl;

					chunkGrid.streamInToGPUChunkNeighborhood(chunk, 1);

					const vec3f& chunkCenter = chunkGrid.getWorldPosChunk(chunk);
					const vec3f& voxelExtends = chunkGrid.getVoxelExtends();
					float virtualVoxelSize = chunkGrid.getHashParams().m_virtualVoxelSize;

					vec3f minCorner = chunkCenter - voxelExtends / 2.0f - vec3f(virtualVoxelSize, virtualVoxelSize, virtualVoxelSize)*(float)chunkGrid.getHashParams().m_SDFBlockSize;
					vec3f maxCorner = chunkCenter + voxelExtends / 2.0f + vec3f(virtualVoxelSize, virtualVoxelSize, virtualVoxelSize)*(float)chunkGrid.getHashParams().m_SDFBlockSize;

					extractIsoSurface(chunkGrid.getHashData(), chunkGrid.getHashParams(), rayCastData, texPoolData, texPoolParams, minCorner, maxCorner, true);

					chunkGrid.streamOutToCPUAll();
				}
			}
		}
	}

	unsigned int nStreamedBlocks;
	chunkGrid.streamInToGPUAll(camPos, radius, true, nStreamedBlocks);

	chunkGrid.startMultiThreading();
}

void CUDAMarchingCubesHashSDF::extractIsoSurface(const HashData& hashData, const HashParams& hashParams, const RayCastData& rayCastData, const TexPoolData& texPoolData, const TexPoolParams& texPoolParams, const vec3f& minCorner, const vec3f& maxCorner, bool boxEnabled)
{
	resetMarchingCubesCUDA(m_data);

	m_params.m_maxCorner = MatrixConversion::toCUDA(maxCorner);
	m_params.m_minCorner = MatrixConversion::toCUDA(minCorner);
	m_params.m_boxEnabled = boxEnabled;
	m_data.updateParams(m_params);


	//extractIsoSurfaceCUDA(hashData, rayCastData, m_params, m_data);		//OLD one-pass version (it's inefficient though)

	extractIsoSurfacePass1CUDA(hashData, rayCastData, m_params, m_data);
	extractIsoSurfacePass2CUDA(hashData, rayCastData, m_params, m_data, texPoolData, m_data.getNumOccupiedBlocks());

	std::cout << "Isosurface finish\n" << std::endl;
	copyTrianglesToCPU(texPoolData, texPoolParams);
}

/*
void CUDAMarchingCubesHashSDF::extractIsoSurfaceCPU(const HashData& hashData, const HashParams& hashParams, const RayCastData& rayCastData)
{
	reset();
	m_params.m_numOccupiedSDFBlocks = hashParams.m_numOccupiedBlocks;
	m_data.updateParams(m_params);

	MarchingCubesData cpuData = m_data.copyToCPU();
	HashData		  cpuHashData = hashData.copyToCPU();

	for (unsigned int sdfBlockId = 0; sdfBlockId < m_params.m_numOccupiedSDFBlocks; sdfBlockId++) {
		for (int x = 0; x < hashParams.m_SDFBlockSize; x++) {
			for (int y = 0; y < hashParams.m_SDFBlockSize; y++) {
				for (int z = 0; z < hashParams.m_SDFBlockSize; z++) {

					const HashEntry& entry = cpuHashData.d_hashCompactified[sdfBlockId];
					if (entry.ptr != FREE_ENTRY) {
						int3 pi_base = cpuHashData.SDFBlockToVirtualVoxelPos(entry.pos);
						int3 pi = pi_base + make_int3(x,y,z);
						float3 worldPos = cpuHashData.virtualVoxelPosToWorld(pi);

						cpuData.extractIsoSurfaceAtPosition(worldPos, cpuHashData, rayCastData);
					}

				} // z
			} // y
		} // x
	} // sdf block id

	// save mesh
	{
		std::cout << "saving mesh..." << std::endl;
		std::string filename = "Scans/scan.ply";
		unsigned int nTriangles = *cpuData.d_numTriangles;

		std::cout << "marching cubes: #triangles = " << nTriangles << std::endl;

		if (nTriangles == 0) return;

		unsigned int baseIdx = (unsigned int)m_meshData.m_Vertices.size();
		m_meshData.m_Vertices.resize(baseIdx + 3*nTriangles);
		m_meshData.m_Colors.resize(baseIdx + 3*nTriangles);

		vec3f* vc = (vec3f*)cpuData.d_triangles;
		for (unsigned int i = 0; i < 3*nTriangles; i++) {
			m_meshData.m_Vertices[baseIdx + i] = vc[2*i+0];
			m_meshData.m_Colors[baseIdx + i] = vc[2*i+1];
		}

		//create index buffer (required for merging the triangle soup)
		m_meshData.m_FaceIndicesVertices.resize(nTriangles);
		for (unsigned int i = 0; i < nTriangles; i++) {
			m_meshData.m_FaceIndicesVertices[i][0] = 3*i+0;
			m_meshData.m_FaceIndicesVertices[i][1] = 3*i+1;
			m_meshData.m_FaceIndicesVertices[i][2] = 3*i+2;
		}

		//m_meshData.removeDuplicateVertices();
		//m_meshData.mergeCloseVertices(0.00001f);
		std::cout << "merging close vertices... ";
		m_meshData.mergeCloseVertices(0.00001f, true);
		std::cout << "done!" << std::endl;
		std::cout << "removing duplicate faces... ";
		m_meshData.removeDuplicateFaces();
		std::cout << "done!" << std::endl;

		std::cout << "saving mesh (" << filename << ") ...";
		MeshIOf::saveToFile(filename, m_meshData);
		std::cout << "done!" << std::endl;

		clearMeshBuffer();
	}

	cpuData.free();
}
*/
