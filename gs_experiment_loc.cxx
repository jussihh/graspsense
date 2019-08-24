
#include "gsDepthCamera.hxx"
#include "gsIRCamera.hxx"
#include "gsTimer.hxx"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include <vector>
#include <chrono>
#include <string>
#include <thread>      
#include <list>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <Windows.h> 

#define PHOXI_OPENCV_SUPPORT
#include "PhoXi.h"
#include "PhoLocalization.h"

struct window {
	std::string name;
	int width, height;
};


void cvTileWindows(std::vector<window> windows) {

	const int x_max = GetSystemMetrics(SM_CXSCREEN);
	const int y_max = GetSystemMetrics(SM_CYSCREEN);

	int x = 0, y = 0;
	int max_height = 0;

	for (const window window_instance : windows) {

		if ((x + window_instance.width) > x_max) {
			x = 0;
			y = y + max_height;
			max_height = window_instance.height;
		}
		else {
			if (window_instance.height > max_height)
				max_height = window_instance.height;
		}

		if (y > y_max)
			std::cout << "Window clipped by screen border." << std::endl;

		cv::moveWindow(window_instance.name, x, y);
		x = x + window_instance.width;
	}
}

void getKinectImages(std::shared_ptr<gsDepthCamera> cam, std::deque<cv::Mat> &ir, std::deque<cv::Mat> &depth, std::deque<cv::Mat> &color, std::mutex &mymutex) {
	cv::Mat ir_tmp, depth_tmp, color_tmp;
	
	cam->clearBuffer();
	
	while (cam->state() != gsDepthCamera::State::DELETED) {
		if (!cam->getImages(ir_tmp, depth_tmp, color_tmp, false)) {
			printf("Cannot get frames. Image getter thread exiting.\n");
			break;
		}
		ir_tmp.convertTo(ir_tmp, CV_8UC1, 255);
		depth_tmp.convertTo(depth_tmp, CV_8UC1, 255);
		{
			std::lock_guard<std::mutex> lock(mymutex);

			ir.clear();
			depth.clear();
			color.clear();
			ir.push_back(ir_tmp);
			depth.push_back(depth_tmp);
			color.push_back(color_tmp);
		}
	}
}

void stopRecordingKinect(std::shared_ptr<gsDepthCamera> cam, std::string experimentname) {
	std::string foldername = experimentname + "/" + cam->getSerial();
	boost::filesystem::create_directory(foldername);
	cam->stopRecordingAndSave(foldername);
}

void snapKinect(std::shared_ptr<gsDepthCamera> cam, std::string foldername) {
	cam->snapshot(foldername);
}

void startIRCamera(unique_ptr<gsIRCamera> &cam) {
	cam->initialize();
	cam->startStreaming();
}

void getIRImages(std::unique_ptr<gsIRCamera> &cam, cv::Mat &image) {
	cv::Mat lCvImage;
	cv::Mat falseColorImage;
	cam->getImage(lCvImage);
	double mini, maxi;
	cv::minMaxLoc(lCvImage, &mini, &maxi);
	lCvImage -= mini;
	lCvImage *= (65536.0 / (maxi - mini));
	lCvImage.convertTo(falseColorImage, CV_8UC3, 1.0 / 255);
	cv::applyColorMap(falseColorImage, falseColorImage, cv::COLORMAP_JET);
	image = falseColorImage;
}

void stopRecordingIR(unique_ptr<gsIRCamera> &cam, std::string experimentname) {
	std::string foldername = experimentname + "/" + cam->getID();
	boost::filesystem::create_directory(foldername);
	cam->stopRecordingAndSave(foldername);
}

void snapIR(std::unique_ptr<gsIRCamera> &cam, std::string foldername) {
	cam->snapshot(foldername);
}


bool snapPhoxi(pho::api::PPhoXi &PhoXiDevice, std::string foldername) {

	std::cout << "Snapping on phoxi" << std::endl;
	if (PhoXiDevice->isAcquiring()) {

		int FrameID = PhoXiDevice->TriggerFrame();
		pho::api::PFrame Frame;
		
		if (FrameID < 0) {
			std::cout << "Trigger was unsuccessful!" << std::endl;
			Frame = nullptr;
		}
		else {
			std::cout << "Frame was triggered, Frame Id: " << FrameID << std::endl;
			std::cout << "Waiting for frame" << std::endl;
			Frame = PhoXiDevice->GetSpecificFrame(FrameID, pho::api::PhoXiTimeout::Infinity);
		}

		std::string PhoxiID = PhoXiDevice->HardwareIdentification.GetValue();
		if (Frame) {
			if (!Frame->Empty()) {

				boost::filesystem::create_directory(foldername);

				if (!Frame->PointCloud.Empty()) {
					Frame->SaveAsPly(foldername + "/" + "cloud.ply");
				}
				if (!Frame->NormalMap.Empty()) {
					cv::Mat normalmap;
					Frame->NormalMap.ConvertTo(normalmap);
					cv::imwrite(foldername + "/" + "normalmap.exr", normalmap);
				}
				if (!Frame->ConfidenceMap.Empty()) {
					cv::Mat confidencemap;
					Frame->ConfidenceMap.ConvertTo(confidencemap);
					cv::imwrite(foldername + "/" + "confidencemap.exr", confidencemap);
				}
				if (!Frame->DepthMap.Empty()) {
					cv::Mat depthmap;
					Frame->DepthMap.ConvertTo(depthmap);
					cv::imwrite(foldername + "/" + "depthmap.exr", depthmap);
				}
				if (!Frame->Texture.Empty()) {
					cv::Mat texturemap;
					Frame->Texture.ConvertTo(texturemap);
					cv::imwrite(foldername + "/" + "texturemap.exr", texturemap);
				}
				return true;
			}
			else {
				std::cout << "Frame is empty." << std::endl;;
			}
		}
		else {
			std::cout << "Failed to retrieve the frame!" << std::endl;;
		}

	}
	else {
		std::cout << "Phoxi is not acquiring." << std::endl;
	}
	return false;
}

bool phoxiLocalize(std::string PhoxiID, std::string localizationconfigpath, std::string storepath) {

	using namespace pho::sdk;
	std::unique_ptr<PhoLocalization> Localization;
	try {
		Localization.reset(new PhoLocalization());
	}
	catch (AuthenticationException ex) {
		std::cout << ex.what() << std::endl;
		return -1;
	}
	SceneSource  Scene = SceneSource::PhoXi(PhoxiID);
	Localization->LoadLocalizationConfiguration(localizationconfigpath);
	Localization->SetSceneSource(Scene);

	AsynchroneResultQueue AsyncQueue = Localization->StartAsync();
	TransformationMatrix4x4 Transform;
	LocalizationPose pose;
	cv::FileStorage fs(storepath, cv::FileStorage::WRITE);
	int objid = 0;

	while (AsyncQueue.GetNext(pose)) {
		cout << pose.Transformation << endl;
		cv::Mat Transformation_cv(4,4, CV_32F, &pose.Transformation);
		fs << "transform_" + std::to_string(objid) << Transformation_cv;
		fs << "visibleoverlap_" + std::to_string(objid) << pose.VisibleOverlap;
		objid++;
	}
	fs << "n_obj" << objid;

	fs.release();

}



int main(int argc, char * argv[])
{

	std::string experimentname = "recording_default"; 
	std::string cameraconfigfilename = "cameraconfig.xml";
	std::string localizationconfigpath = "localization.plcf";

	boost::program_options::options_description desc{ "Options" };
	std::vector<std::string> infiles;
	desc.add_options()
		("help,h", "Help screen")
		("output,o", boost::program_options::value<std::string>(), "Output folder")
		("localization,l", boost::program_options::value<std::string>(), "Localization config file")
		("cameraconfig,c", boost::program_options::value<std::string>(), "Camera configuration file (FileStorage xml/yaml)");


	boost::program_options::variables_map vm;
	boost::program_options::store(parse_command_line(argc, argv, desc), vm);
	boost::program_options::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << '\n';
		return(0);
	}

	if (vm.count("output")) {
		experimentname = vm["output"].as<std::string>();
	}

	if (vm.count("cameraconfig")) {
		cameraconfigfilename = vm["cameraconfig"].as<std::string>();
	}

	if (vm.count("localization")) {
		localizationconfigpath= vm["localization"].as<std::string>();
	}

	std::vector<std::string> ircameraIDs, kinectSerials, liveKinectSerials;
	std::vector<int> liveIRcameras, liveKinects;
	std::string PhoxiID = "";

	cv::FileStorage fs(cameraconfigfilename, cv::FileStorage::READ);	
	fs["kinectSerials"] >> kinectSerials;
	fs["liveKinects"] >> liveKinects;
	fs["ircameraIDs"] >> ircameraIDs;
	fs["liveIRcameras"] >> liveIRcameras;
	fs["PhoxiID"] >> PhoxiID;
	fs.release();


	for(std::string s:kinectSerials)
		std::cout << s <<"\n";
	std::cout << "of which live: \n";
	for (int i:liveKinects) {
		liveKinectSerials.push_back(kinectSerials[i]);
		std::cout << kinectSerials[i] << "\n";
	}



	std::vector<std::unique_ptr<gsIRCamera>> lgsIRCameras;
	 
	for (std::string id : ircameraIDs) {
		lgsIRCameras.push_back(std::make_unique<gsIRCamera>(id));
	}

	std::vector<std::thread> startThreads;
	for (unique_ptr<gsIRCamera> &lgsIRCamera : lgsIRCameras) {
		startThreads.push_back(std::thread(startIRCamera, std::ref(lgsIRCamera)));
	}

	for (int i = 0; i < startThreads.size(); i++) {
		startThreads[i].join();
	}

	std::vector<cv::Mat> images(lgsIRCameras.size());
	std::vector<std::thread> getIRThreads;

	std::vector<window> windows;

	for (int i = 0; i < lgsIRCameras.size(); i++) {
		std::string windowname = "Flir " + lgsIRCameras[i]->getID();
		cv::namedWindow(windowname);
		windows.push_back({ windowname, 640, 512 });
	}
	


	#ifdef WITH_CUDA
		gsDepthCamera::Processor freenectprocessor = gsDepthCamera::Processor::CUDA;
	#else
		gsDepthCamera::Processor freenectprocessor = gsDepthCamera::Processor::OPENGL;
	#endif
	std::vector<std::shared_ptr<gsDepthCamera>> depthCameras;


	for (int i = 0; i < kinectSerials.size(); i++) {
		depthCameras.push_back(std::make_shared<gsDepthCamera>(freenectprocessor, kinectSerials[i]));
	}

	int ndepthcam = static_cast<int>(depthCameras.size());
	std::vector<std::deque<cv::Mat>> color(ndepthcam, std::deque<cv::Mat>(1)), depth(ndepthcam, std::deque<cv::Mat>(1)), ir(ndepthcam, std::deque<cv::Mat>(1));

	std::vector<std::mutex> mutexes(ndepthcam);

	for (int i = 0; i < ndepthcam; i++) {
		depthCameras[i]->getImages(ir[i][0], depth[i][0], color[i][0], false);
		std::string dwname = std::string("Kinect " + depthCameras[i]->getSerial() + ": Depth");
		std::string cwname = std::string("Kinect " + depthCameras[i]->getSerial() + ": Color");
		std::string iwname = std::string("Kinect " + depthCameras[i]->getSerial() + ": Ir");
	
		cv::namedWindow(dwname);
		cv::namedWindow(cwname, cv::WINDOW_NORMAL);
		cv::namedWindow(iwname);

		windows.push_back({ dwname, depth[i][0].cols, depth[i][0].rows });
		windows.push_back({ cwname, color[i][0].cols / 2, color[i][0].rows / 2 });
		windows.push_back({ iwname, ir[i][0].cols, ir[i][0].rows });
	}


	std::vector<std::thread> getDepthThreads;

	for (int i = 0; i < ndepthcam; i++) {
		getDepthThreads.push_back(std::thread(getKinectImages, depthCameras[i], std::ref(ir[i]), std::ref(depth[i]), std::ref(color[i]), std::ref(mutexes[i])));
	}

	
	for (int i = 0; i < ndepthcam; i++) {
		if (std::find(liveKinects.begin(), liveKinects.end(), i) != liveKinects.end()) {
			
		}
		else {
			depthCameras[i]->pause();
		}
	}


	enum status { DISPLAY, RECORD, SAVE, SNAPSHOT, QUIT, PAUSE };
	status viewer_status = DISPLAY;
	uint64_t recstart, recend;

	int iteration = 0;
	std::string experimentname_base = experimentname;


	while (viewer_status != QUIT) {

		/*************
			Kinects
		*/
		if (viewer_status == DISPLAY) {
			for (int i = 0; i < ndepthcam; i++) {
				std::lock_guard<std::mutex> lock(mutexes[i]);
				if (!depth[i].empty()) {
					cv::imshow("Kinect " + depthCameras[i]->getSerial() + ": Depth", depth[i].front());
					depth[i].pop_front();
				}
				if (!ir[i].empty()) {
					cv::imshow("Kinect " + depthCameras[i]->getSerial() + ": Ir", ir[i].front());
					ir[i].pop_front();
				}
				if (!color[i].empty()) {
					cv::imshow("Kinect " + depthCameras[i]->getSerial() + ": Color", color[i].front());
					cv::resizeWindow("Kinect " + depthCameras[i]->getSerial() + ": Color", color[i].front().cols / 2, color[i].front().rows / 2);
					color[i].pop_front();
				}
			}
		}
		/*************
			Flirs
		*/
		if (viewer_status == DISPLAY) {
			for (int i = 0; i < images.size(); i++) {
				getIRThreads.push_back(std::thread(getIRImages, std::ref(lgsIRCameras[i]), std::ref(images[i])));
			}
			for (int i = 0; i < getIRThreads.size(); i++) {
				if (getIRThreads[i].joinable())
					getIRThreads[i].join();
			}
			for (int i = 0; i < images.size(); i++) {
				imshow("Flir " + lgsIRCameras[i]->getID(), images[i]);
			}
		}
		/*************
			UI
		*/

		cvTileWindows(windows);

		int c = cv::waitKey(1);
		
		switch(c){
			case (int)'q':

				viewer_status = QUIT;
				break;
			case 32:
				std::cout << "Pressed SPACE" << std::endl;
				if (viewer_status == DISPLAY)
					viewer_status = SNAPSHOT;
				else
					std::cout << "Cannot take a snapshot while paused. Use window toolbar to save.\n";
				break;
			case (int)'r': 

				if (viewer_status == DISPLAY) {
					Beep(494 * 2, 100); Sleep(50);
					Beep(330 * 2, 100);
					viewer_status = RECORD;
					recstart = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
					for (unique_ptr<gsIRCamera> &lgsIRCamera : lgsIRCameras) {
						lgsIRCamera->startRecording();
					}
					for (shared_ptr<gsDepthCamera> &lgsDepthCamera : depthCameras) {
						lgsDepthCamera->startRecording();
					}

					std::cout << "Recording started" << std::endl;
				}
				break;
			case (int)'p':
				if (viewer_status == DISPLAY) {
					Beep(330 * 2, 100); 
					for (shared_ptr<gsDepthCamera> &lgsDepthCamera : depthCameras) {
						lgsDepthCamera->pause();
						viewer_status = PAUSE;
					}
				}
				else if (viewer_status == PAUSE) {
					Beep(494 * 2, 100); 
					for (shared_ptr<gsDepthCamera> &lgsDepthCamera : depthCameras) {
						if (std::find(liveKinectSerials.begin(), liveKinectSerials.end(), lgsDepthCamera->getSerial()) != liveKinectSerials.end()) {
							lgsDepthCamera->unpause();
						}
						else {
						}						
						viewer_status = DISPLAY;
					}
				}
				break;

			case (int)'s':

				if (viewer_status == RECORD) {					
					viewer_status = SAVE;
					recend = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
					std::cout << "Pressed s" << std::endl;
					Beep(330 * 2, 100); Sleep(50);
					Beep(494 * 2, 100); 
				}
				break;
			case (int)'n':
				if (viewer_status == DISPLAY) {
					for (unique_ptr<gsIRCamera> &lgsIRCamera : lgsIRCameras) {
						lgsIRCamera->performNUC();
					}
				}
				break;
			default:
				break;

		}

		if (viewer_status == SNAPSHOT) {
			
			std::string snapfolder = "snapshots";
			if(!boost::filesystem::exists(snapfolder))
				boost::filesystem::create_directory(snapfolder);

			if (boost::filesystem::is_directory(snapfolder)) {
				std::vector<std::thread> snapthreads;
				for (unique_ptr<gsIRCamera> &lgsIRCamera : lgsIRCameras) {
					snapthreads.push_back(std::thread(snapIR, std::ref(lgsIRCamera), snapfolder));
				}
				for (shared_ptr<gsDepthCamera> &lgsDepthCamera : depthCameras) {
					if (std::find(liveKinectSerials.begin(), liveKinectSerials.end(), lgsDepthCamera->getSerial()) != liveKinectSerials.end()) {
						snapthreads.push_back(std::thread(snapKinect, std::ref(lgsDepthCamera), snapfolder));
					}
				}
				for (std::thread &sThread : snapthreads) {
					sThread.join();
				}
				std::cout << "Snapshots saved." << std::endl;
			}
			else {
				std::cout << "Could not create folder for snapshots!" << std::endl;
			}
			viewer_status = DISPLAY;

		}

		if (viewer_status == SAVE) {

			experimentname = experimentname_base + "_" + std::to_string(iteration);
			boost::filesystem::create_directory(experimentname);

			std::vector<std::thread> savingThreads;

			for (unique_ptr<gsIRCamera> &lgsIRCamera : lgsIRCameras) {
				savingThreads.push_back(std::thread(stopRecordingIR, std::ref(lgsIRCamera), experimentname));
			}
			for (shared_ptr<gsDepthCamera> &lgsDepthCamera : depthCameras) {
				savingThreads.push_back(std::thread(stopRecordingKinect, std::ref(lgsDepthCamera), experimentname));
			}

			for (std::thread &sThread:savingThreads) {
					sThread.join();
			}

			for (std::thread &gThread : getDepthThreads) {
				gThread.join();
			}


			getDepthThreads.clear();


			for (shared_ptr<gsDepthCamera> &lgsDepthCamera : depthCameras) {
				lgsDepthCamera->pause();
			}

			std::string staticfoldername = experimentname + "/static";
			boost::filesystem::create_directory(staticfoldername);

			phoxiLocalize(PhoxiID, localizationconfigpath, experimentname + "/localization.yaml");
			for (int i = 0; i < ndepthcam; i++) {
				if (std::find(liveKinects.begin(), liveKinects.end(), i) != liveKinects.end()) {
					depthCameras[i]->unpause();
				}
				else {
				}
			}
			
			for (int i = 0; i < ndepthcam; i++) {
				getDepthThreads.push_back(std::thread(getKinectImages, depthCameras[i], std::ref(ir[i]), std::ref(depth[i]), std::ref(color[i]), std::ref(mutexes[i])));
			}
			
			cv::FileStorage fs(experimentname + "/experiment.xml", cv::FileStorage::WRITE);
			fs << "ircameraIDs" << ircameraIDs;
			fs << "kinectSerials" << kinectSerials;
			fs << "liveIRcameras" << liveIRcameras;
			fs << "liveKinects" << liveKinects;
			fs << "PhoxiID" << PhoxiID;
			fs << "recstart" << static_cast<double>(recstart);
			fs << "recend" << static_cast<double>(recend);
			fs.release();
			viewer_status = DISPLAY;
			Beep(330*2, 100); Sleep(50);	
			Beep(392*2, 100); Sleep(50);
			Beep(330*2, 100); Sleep(50);
			Beep(261*2, 100); Sleep(50);
			Beep(294*2, 100); Sleep(50);
			Beep(392*2, 100);
			iteration++;
		}

		if(viewer_status == QUIT)
			break;

	}

	/*************
		Cleanup
	*/

	for (auto& depthcam : depthCameras) {
		depthcam->unpause();
		depthcam->shutDown();
	}
	for (unique_ptr<gsIRCamera> &lgsIRCamera : lgsIRCameras) {
		lgsIRCamera->stopStreaming();
	}

	for (int i = 0; i < ndepthcam; i++) {
		if (getDepthThreads[i].joinable())
			getDepthThreads[i].join();
	}

	std::cout << "Exiting.\n";
	return 1;

}
