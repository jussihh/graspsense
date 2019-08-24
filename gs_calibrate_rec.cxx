
#include "gsDepthCamera.hxx"
#include "gsIRCamera.hxx"
#include "gsTimer.hxx"


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ccalib/multicalib.hpp>


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

void snapKinect(std::shared_ptr<gsDepthCamera> cam, std::string foldername, std::vector<std::string> filenames) {
	cam->snapshot(foldername, true, filenames);
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

void snapIR(std::unique_ptr<gsIRCamera> &cam, std::string foldername, std::string filename) {
	cam->snapshot(foldername, false, filename);
}

struct combination {
	std::vector<std::string> flirs, kinects, kinectframes, phoxis; 
	int n;
	bool find(std::string id) {
		return (std::find(flirs.begin(), flirs.end(), id) != flirs.end()) || 
			   (std::find(kinects.begin(), kinects.end(), id) != kinects.end()) ||
				(std::find(phoxis.begin(), phoxis.end(), id) != phoxis.end());
	}
	friend std::ostream& operator<< (std::ostream& stream, const combination& acombination) {
		for (int i = 0; i < acombination.flirs.size(); i++)
			stream << " " << acombination.flirs[i];
		for (int i = 0; i < acombination.kinects.size(); i++)
			stream << " " << acombination.kinects[i];
		for (int i = 0; i < acombination.kinectframes.size(); i++)
			stream << " " << acombination.kinectframes[i];
		for (int i = 0; i < acombination.phoxis.size(); i++)
			stream << " " << acombination.phoxis[i];
		stream << " x " << acombination.n;
		stream << std::endl;
		return stream;
	}
};

struct calibrationset {
	std::string name;
	std::string type;
	std::string pattern;
	int n;
	std::vector<std::string> cameramapping;
	std::vector<combination> combinations;

	friend std::ostream& operator<< (std::ostream& stream, const calibrationset& acalibrationset) {
		stream << "Name: " << acalibrationset.name << std::endl;
		stream << "Type: " << acalibrationset.type << ", Pattern: " << acalibrationset.pattern << ", N: " << acalibrationset.n << std::endl;
		for (int i = 0; i < acalibrationset.combinations.size(); i++)
			stream << acalibrationset.combinations[i];
		stream << std::endl;
		return stream;
	}
};


int cam_str2int(std::vector<std::string> cameramapping, std::string id) {
	std::vector<std::string>::iterator it = find(cameramapping.begin(), cameramapping.end(), id);
	if (it != cameramapping.end())
		return (it - cameramapping.begin());
	else
		return -1;
}


std::string cam_int2str(std::vector<std::string> cameramapping, int index) {
	if (index >= 0 && index < cameramapping.size())
		return cameramapping[index];
	else
		return "";
}

bool snapPhoxi(pho::api::PPhoXi &PhoXiDevice, cv::Mat &texturemap, cv::Mat &depthmap){

	if (PhoXiDevice->isAcquiring()) {

		pho::api::PFrame Frame = PhoXiDevice->GetFrame(pho::api::PhoXiTimeout::Infinity);
		if (Frame) {
			if (!Frame->Empty()) {
				if (!Frame->DepthMap.Empty()) {
					Frame->DepthMap.ConvertTo(depthmap);
				}
				if (!Frame->Texture.Empty()) {
					Frame->Texture.ConvertTo(texturemap);
				}
				return true;
			}
			else {
				std::cout << "Frame is empty.";
			}
		}
		else {
			std::cout << "Failed to retrieve the frame!";
		}

	}
	return false;
}



int main(int argc, char * argv[])
{

	pho::api::PhoXiFactory Factory;
	std::cout << "Phoxi control is " << (Factory.isPhoXiControlRunning() ? "" : " not ") << "running.";
	std::string PhoxiID = "1801020C3";
	pho::api::PhoXiTimeout Timeout = pho::api::PhoXiTimeout::ZeroTimeout;
	pho::api::PPhoXi PhoXiDevice = Factory.CreateAndConnect(PhoxiID, Timeout);
	if (PhoXiDevice) {
		std::cout << "Connection to the device " << PhoxiID  << " was Successful!" << std::endl;
	}
	else {
		std::cout << "Connection to the device " << PhoxiID << " was Unsuccessful!" << std::endl;
	}
	if (PhoXiDevice->isAcquiring()) {
		PhoXiDevice->StopAcquisition();
	}
	std::cout << "Starting Freerun mode" << std::endl;
	PhoXiDevice->TriggerMode = pho::api::PhoXiTriggerMode::Freerun;
	PhoXiDevice->ClearBuffer();
	

	cv::Mat phoxi_texturemap, phoxi_depthmap;
	bool all_sets = true;
	int selected_set = -1;

	boost::program_options::options_description desc{ "Options" };
	std::vector<std::string> infiles;
	desc.add_options()
		("help,h", "Help screen")
		("set,s", boost::program_options::value<int>(), "Set number (0,1,...)");

	boost::program_options::variables_map vm;
	boost::program_options::store(parse_command_line(argc, argv, desc), vm);
	boost::program_options::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << '\n';
		return(0);
	}

	if (vm.count("set")) {
		all_sets = false;
		selected_set = vm["set"].as<int>();
	}

	cv::FileStorage fs2("calibrationconfig.xml", cv::FileStorage::READ);

	cv::FileNode calibrationsets = fs2["calibrationsets"];
	int idx = 0;
	std::vector<std::string> kinects, frames, flirs;
	std::vector<calibrationset> csets;

	for (cv::FileNode cset : calibrationsets) {
		csets.push_back(calibrationset());
		csets.back().name = (std::string)cset["name"];
		csets.back().type = (std::string)cset["type"];
		csets.back().pattern = (std::string)cset["pattern"];
		csets.back().n = (int)cset["n"];
		csets.back().cameramapping = std::vector<std::string>();

		cv::FileNode cameramappingnode = cset["cameramapping"];
		for (cv::FileNode camera : cameramappingnode) {
			csets.back().cameramapping.push_back((std::string)camera);
		}

		cv::FileNode combinations = cset["combinations"];
		for (cv::FileNode combinationfn : combinations)
		{
			csets.back().combinations.push_back(combination());
			combinationfn["flirs"] >> csets.back().combinations.back().flirs;
			combinationfn["kinects"] >> csets.back().combinations.back().kinects;
			combinationfn["kinectframes"] >> csets.back().combinations.back().kinectframes;
			combinationfn["phoxis"] >> csets.back().combinations.back().phoxis;
			if ((int)combinationfn["n"])
				csets.back().combinations.back().n = (int)combinationfn["n"];
			else
				csets.back().combinations.back().n = (int)cset["n"];
		}
		cout << csets.back();
	}

	fs2.release();


	std::vector<std::unique_ptr<gsIRCamera>> lgsIRCameras;
	std::vector<std::string> ircameraIDs = { "192.168.1.101", "192.168.1.102","192.168.1.103","192.168.1.104" };
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

	std::vector<std::string> kinectSerials = { "003817260847", "003643560847", "000807660847", "000785160847" };

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

	enum status { DISPLAY, RECORD, SAVE, SNAPSHOT, QUIT };
	status viewer_status = DISPLAY;

	
	int set_i = 0;
	if (!all_sets && selected_set < csets.size()) {
		set_i = selected_set;
	}

	int combo_i = 0;
	int repeat_i = 0;
	int snap_i = 0;
	cv::Mat empty(cv::Size(1, 1), CV_8UC1, cv::Scalar(0));
	std::vector<std::string> filelist;
	filelist.push_back(std::string("pattern.png"));

	if (csets[set_i].combinations[combo_i].find(PhoxiID)) {
		PhoXiDevice->StartAcquisition();
	}


	for (shared_ptr<gsDepthCamera> &lgsDepthCamera : depthCameras) {
		bool found = csets[set_i].combinations[combo_i].find(lgsDepthCamera->getSerial());
		if (found) {
			lgsDepthCamera->unpause();
		}
		else{
			lgsDepthCamera->pause();
		}
	}

	while (viewer_status != QUIT) {

		if (viewer_status == DISPLAY) {
			for (int i = 0; i < ndepthcam; i++) {
				std::lock_guard<std::mutex> lock(mutexes[i]);
				bool found = csets[set_i].combinations[combo_i].find(depthCameras[i]->getSerial());
				if (!depth[i].empty()) {					
					cv::imshow("Kinect " + depthCameras[i]->getSerial() + ": Depth", found ? depth[i].front() : empty);
					depth[i].pop_front();
				}
				if (!ir[i].empty()) {
					cv::imshow("Kinect " + depthCameras[i]->getSerial() + ": Ir", found ? ir[i].front() : empty);
					ir[i].pop_front();
				}
				if (!color[i].empty()) {
					std::string wname = "Kinect " + depthCameras[i]->getSerial() + ": Color";
					cv::imshow(wname, found ? color[i].front() : empty);
					if(found) 
						cv::resizeWindow(wname, color[i].front().cols / 2, color[i].front().rows / 2);
					color[i].pop_front();
				}
			}
		}

		if (viewer_status == DISPLAY) {
			for (int i = 0; i < images.size(); i++) {
				getIRThreads.push_back(std::thread(getIRImages, std::ref(lgsIRCameras[i]), std::ref(images[i])));
			}
			for (int i = 0; i < getIRThreads.size(); i++) {
				if (getIRThreads[i].joinable())
					getIRThreads[i].join();
			}
			for (int i = 0; i < images.size(); i++) {
				bool found = csets[set_i].combinations[combo_i].find(lgsIRCameras[i]->getID());
				imshow("Flir " + lgsIRCameras[i]->getID(), found ? images[i] : empty);
			}
		}

		if (csets[set_i].combinations[combo_i].find(PhoxiID)) {
			if (snapPhoxi(PhoXiDevice, phoxi_texturemap, phoxi_depthmap)) {
				cv::imshow("PhoXi depth map", phoxi_depthmap / 1000);
				cv::imshow("Phoxi texture map", phoxi_texturemap / 1000);
			}
			else {
				std::cout << "Failed to capture on Phoxi." << std::endl;
			}
		}

		cvTileWindows(windows);

		int c = cv::waitKey(1);
		
		switch(c){
			case (int)'q':

				viewer_status = QUIT;
				break;
			case 32: 
				viewer_status = SNAPSHOT;
				std::cout << "Pressed SPACE" << std::endl;
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
			std::cout << "Snap id: " << snap_i << " repeat: " << repeat_i << " combo: " << combo_i << " set: " << set_i << std::endl;
			std::string snapfolder = "calibration_set_" + std::to_string(set_i);

			if(!boost::filesystem::exists(snapfolder))
				boost::filesystem::create_directory(snapfolder);

			if (boost::filesystem::is_directory(snapfolder)) {
				std::vector<std::thread> snapthreads;
				for (shared_ptr<gsDepthCamera> &lgsDepthCamera : depthCameras) {
					bool found = csets[set_i].combinations[combo_i].find(lgsDepthCamera->getSerial());
					if (found) {
						int camid = cam_str2int(csets[set_i].cameramapping, lgsDepthCamera->getSerial());
						std::string nirfname = std::to_string(camid) + "-" + std::to_string(snap_i) + ".jpg";
						std::string depthfname = std::to_string(camid) + "-" + std::to_string(snap_i) + "_depth.exr";
						std::string colorfname = std::to_string(camid + 1) + "-" + std::to_string(snap_i) + ".jpg";

						if (!std::any_of(csets[set_i].combinations[combo_i].kinectframes.begin(), csets[set_i].combinations[combo_i].kinectframes.end(), [](std::string s) {return s == "Depth"; }))
							depthfname = "";

						if (!std::any_of(csets[set_i].combinations[combo_i].kinectframes.begin(), csets[set_i].combinations[combo_i].kinectframes.end(), [](std::string s) {return s=="IR"; }))
							nirfname = "";
						else
							filelist.push_back(nirfname);			

						if (!std::any_of(csets[set_i].combinations[combo_i].kinectframes.begin(), csets[set_i].combinations[combo_i].kinectframes.end(), [](std::string s) {return s == "Color"; }))
							colorfname = "";
						else
							filelist.push_back(colorfname);


						std::vector<std::string> fnames = {
							nirfname, 
							depthfname, 
							colorfname 
						};
						snapthreads.push_back(std::thread(snapKinect, std::ref(lgsDepthCamera), snapfolder, fnames));
					}
				}
				for (unique_ptr<gsIRCamera> &lgsIRCamera : lgsIRCameras) {
					bool found = csets[set_i].combinations[combo_i].find(lgsIRCamera->getID());
					if (found) {
						int camid = cam_str2int(csets[set_i].cameramapping, lgsIRCamera->getID());
						if (camid >= 0) {
							std::string irfname = std::to_string(camid) + "-" + std::to_string(snap_i) + ".exr";
							snapthreads.push_back(std::thread(snapIR, std::ref(lgsIRCamera), snapfolder, irfname));
							filelist.push_back(irfname);
						}
					}
				}
				for (std::thread &sThread : snapthreads) {
					sThread.join();
				}

				if (csets[set_i].combinations[combo_i].find(PhoxiID)) {
					int camid = cam_str2int(csets[set_i].cameramapping, PhoxiID);
					std::string depthmapname = std::to_string(camid) + "-" + std::to_string(snap_i) + "_depth.exr";
					std::string texturename = std::to_string(camid) + "-" + std::to_string(snap_i) + ".exr";
					cv::imwrite(snapfolder + "/" + depthmapname, phoxi_depthmap);
					cv::imwrite(snapfolder + "/" + texturename, phoxi_texturemap);
					filelist.push_back(texturename);

				}

				std::cout << "Snapshots saved." << std::endl;
			}
			else {
				std::cout << "Could not create folder for snapshots!" << std::endl;
			}
			viewer_status = DISPLAY;
			
			snap_i++;
			repeat_i++;
			if (repeat_i >= csets[set_i].combinations[combo_i].n) {
				combo_i++;
				repeat_i = 0;
				if (combo_i >= csets[set_i].combinations.size()) {

					for (std::string& sp : filelist) {
						sp = snapfolder + "/" + sp;
					}
					

					cv::FileStorage filelist_fs(snapfolder + "/filelist.xml", cv::FileStorage::WRITE);
					filelist_fs << "filelist" << filelist;
					filelist_fs.release();
					filelist.clear();
					filelist.push_back(std::string("pattern.png"));
					set_i++;
					combo_i = 0;
					snap_i = 0;
					if (set_i >= csets.size() || !all_sets) {
						std::cout << "All calibration sets completed." << std::endl;
						viewer_status = QUIT;
						break;
					}
				}

				if (csets[set_i].combinations[combo_i].find(PhoxiID)) {
					if (!PhoXiDevice->isAcquiring()) {
						PhoXiDevice->ClearBuffer();
						PhoXiDevice->StartAcquisition();
					}
				}
				else {
					if (PhoXiDevice->isAcquiring()) {
						PhoXiDevice->StopAcquisition();
					}
				}

				for (shared_ptr<gsDepthCamera> &lgsDepthCamera : depthCameras) {
					bool found = csets[set_i].combinations[combo_i].find(lgsDepthCamera->getSerial());
					if (found) {
						lgsDepthCamera->unpause();
					}
					else {
						lgsDepthCamera->pause();
					}
				}

			}
		}


		if(viewer_status == QUIT)
			break;


	}

	PhoXiDevice->StopAcquisition();
	PhoXiDevice->Disconnect(true);

	for (unique_ptr<gsIRCamera> &lgsIRCamera : lgsIRCameras) {
		lgsIRCamera->stopStreaming();
	}

	for (auto& depthcam : depthCameras) {
		depthcam->unpause();
		depthcam->shutDown();
	}
	for (int i = 0; i < ndepthcam; i++) {
		if (getDepthThreads[i].joinable())
			getDepthThreads[i].join();
	}


	std::cout << "Exiting.\n";
	return 1;

}
