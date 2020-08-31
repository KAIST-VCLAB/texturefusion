
#include "stdafx.h"
//#include "DepthSensing.h"
#include "texFusion.h"
#include "StructureSensor.h"
#include "SensorDataReader.h"
#include "SensorDataReaderZhou.h"
#include "cudaDebug.h"

//int WINAPI main(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow)
int main(int argc, char** argv)
{
	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

#ifdef OBJECT_SENSING
	ObjectSensing::getInstance()->initQtApp(false);
	ObjectSensing::getInstance()->detach();
#endif // OBJECT_SENSING

	try {
		std::string fileNameDescGlobalApp;
		std::string fileNameDescGlobalTracking;
		std::string fileNameDescGlobalWarping;
		
		if (argc >= 4) {
			fileNameDescGlobalApp = std::string(argv[1]);
			fileNameDescGlobalTracking = std::string(argv[2]);
			fileNameDescGlobalWarping = std::string(argv[3]);
		}
		else {
			std::cout << "usage: DepthSensing [fileNameDescGlobalApp] [fileNameDescGlobalTracking] [fileNameDescGlobalWarping]" << std::endl;
			fileNameDescGlobalApp = "zParametersDefault.txt";
			fileNameDescGlobalTracking = "zParametersTrackingDefault.txt";
			fileNameDescGlobalWarping = "zParametersWarpingDefault.txt";
		}
		std::cout << VAR_NAME(fileNameDescGlobalApp) << " = " << fileNameDescGlobalApp << std::endl;
		std::cout << VAR_NAME(fileNameDescGlobalTracking) << " = " << fileNameDescGlobalTracking << std::endl;
		std::cout << VAR_NAME(fileNameDescGlobalWarping) << " = " << fileNameDescGlobalWarping << std::endl;
		std::cout << std::endl;

		//Read the global app state
		ParameterFile parameterFileGlobalApp(fileNameDescGlobalApp);
		std::ofstream out;

		if (argc >= 6) //for scan net: overwrite .sens file
		{
			for (unsigned int i = 0; i < (unsigned int)argc - 3; i++) {
				const std::string filename = std::string(argv[i+3]);
				const std::string paramName = "s_binaryDumpSensorFile[" + std::to_string(i) + "]";
				parameterFileGlobalApp.overrideParameter(paramName, filename);
				std::cout << "Overwriting s_binaryDumpSensorFile; now set to " << filename << std::endl;

				if (i == 0) {
					//redirect stdout to file
					out.open(util::removeExtensions(filename) + ".voxelhashing.log");
					if (!out.is_open()) throw MLIB_EXCEPTION("unable to open log file " + util::removeExtensions(filename) + ".voxelhashing.log");
					//std::streambuf *coutbuf = std::cout.rdbuf(); //save old buf
					std::cout.rdbuf(out.rdbuf()); //redirect std::cout to out.txt!
				}
			}
		}

		GlobalAppState::getInstance().readMembers(parameterFileGlobalApp);

		//Read the global camera tracking state
		ParameterFile parameterFileGlobalTracking(fileNameDescGlobalTracking);
		GlobalCameraTrackingState::getInstance().readMembers(parameterFileGlobalTracking);
		GlobalCameraTrackingState::getInstance().print();

		//Read the global warping state
		ParameterFile parameterFileGlobalWarping(fileNameDescGlobalWarping);
		GlobalWarpingState::getInstance().readMembers(fileNameDescGlobalWarping);
		GlobalWarpingState::getInstance().print();

		{
			//OVERWRITE streaming radius and streaming pose
			auto& gas = GlobalAppState::get();

			float chunkExt = std::max(std::max(gas.s_streamingVoxelExtents.x, gas.s_streamingVoxelExtents.y), gas.s_streamingVoxelExtents.z);
			float chunkRadius = 0.5f*chunkExt*sqrt(3.0f);

			float frustExt = gas.s_SDFMaxIntegrationDistance - gas.s_sensorDepthMin;
			float frustRadius = 0.5f*frustExt*sqrt(3.0f);	//this assumes that the fov is less than 90 degree

			gas.s_streamingPos = vec3f(0.0f, 0.0f, gas.s_sensorDepthMin + 0.5f*frustExt);
			gas.s_streamingRadius = frustRadius + chunkRadius;

			std::cout << "overwriting s_streamingPos,\t now " << gas.s_streamingPos << std::endl;
			std::cout << "overwriting s_streamingRadius,\t now " << gas.s_streamingRadius << std::endl;

		}
			
		// Set DXUT callbacks
		DXUTSetCallbackDeviceChanging(ModifyDeviceSettings);
		DXUTSetCallbackMsgProc(MsgProc);
		DXUTSetCallbackKeyboard(OnKeyboard);
		DXUTSetCallbackFrameMove(OnFrameMove);

		DXUTSetCallbackD3D11DeviceAcceptable(IsD3D11DeviceAcceptable);
		DXUTSetCallbackD3D11DeviceCreated(OnD3D11CreateDevice);
		DXUTSetCallbackD3D11SwapChainResized(OnD3D11ResizedSwapChain);
		DXUTSetCallbackD3D11FrameRender(OnD3D11FrameRender);
		DXUTSetCallbackD3D11SwapChainReleasing(OnD3D11ReleasingSwapChain);
		DXUTSetCallbackD3D11DeviceDestroyed(OnD3D11DestroyDevice);

		InitApp();
		bool bShowMsgBoxOnError = false;
		DXUTInit(true, bShowMsgBoxOnError); // Parse the command line, show msgboxes on error, and an extra cmd line param to force REF for now
		DXUTSetCursorSettings(true, true); // Show the cursor and clip it when in full screen
		DXUTCreateWindow(GlobalAppState::get().s_windowWidth, GlobalAppState::get().s_windowHeight, L"VoxelHashing", false);

		DXUTSetIsInGammaCorrectMode(false);	//gamma fix (for kinect)

		DXUTCreateDevice(D3D_FEATURE_LEVEL_11_0, true, GlobalAppState::get().s_windowWidth, GlobalAppState::get().s_windowHeight);
		DXUTMainLoop(); // Enter into the DXUT render loop

	}
	catch (const std::exception& e)
	{
		MessageBoxA(NULL, e.what(), "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}
	catch (...)
	{
		MessageBoxA(NULL, "UNKNOWN EXCEPTION", "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}

	//this is a bit of a hack due to a bug in std::thread (a static object cannot join if the main thread exists)
	auto* s = getRGBDSensor();
	SAFE_DELETE(s);

	return DXUTGetExitCode();
}
