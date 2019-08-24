#pragma once
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/logger.h>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <opencv2/core.hpp>
#include <deque>
#include <signal.h>
#include <cstdlib>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>


class gsDepthCamera {

public:
enum Processor { CPU, OPENCL, OPENGL, CUDA, CUDADUMP };
enum State { UNREADY, READY, CLEARING, RECORDING, SAVING, VIRTUAL, DELETED };

private:

	libfreenect2::Freenect2 freenect2_;
	libfreenect2::Freenect2Device * dev_ = NULL;
	libfreenect2::Registration * registration_ = NULL;
	libfreenect2::SyncMultiFrameListener listener_;
	libfreenect2::FrameMap frames_;
	libfreenect2::FrameMap new_frames_;
	libfreenect2::Frame undistorted_, registered_, big_mat_;

	std::string serial_;
	int map_[512 * 424];

	std::deque<libfreenect2::FrameMap> frameBuffer_;
	std::deque<libfreenect2::FrameMap>::iterator virtualCamIterator_;
	const int MAX_BUFFER_SIZE_ = 512; 
	std::mutex getImagesMutex_;
	std::mutex frameMutex_;
	std::mutex bufferHasDataMutex_;
	bool bufferHasData_;
	int n_frames_;
	std::condition_variable bufferHasDataCV_;
	uint64_t recStartTime_, recStopTime_, previousFrameStamp_;
	volatile gsDepthCamera::State state_;

	std::string path_to_folder_;
	std::thread frameThread_;





public:	


	gsDepthCamera::gsDepthCamera(gsDepthCamera::Processor p = CPU, std::string serial = std::string());
	
	gsDepthCamera::gsDepthCamera(std::string path_to_folder);
	gsDepthCamera::~gsDepthCamera();

	bool gsDepthCamera::loadData();
	
	std::vector<std::string> gsDepthCamera::getSerials();
	
	std::string getSerial();


	gsDepthCamera::State gsDepthCamera::state();

	bool gsDepthCamera::seek(uint64_t time_in_microseconds);
	bool gsDepthCamera::getImages(cv::Mat &ir_mat, cv::Mat &depth_mat, cv::Mat &color_mat, bool registered = false, std::vector<uint64_t> *stamps = nullptr);
	void gsDepthCamera::clearBuffer();
	void gsDepthCamera::shutDown();
	void gsDepthCamera::pause();
	void gsDepthCamera::unpause();

	void gsDepthCamera::startRecording();
	void gsDepthCamera::stopRecordingAndSave(std::string foldername);
	
	void gsDepthCamera::snapshot(std::string foldername, bool jpeg = false, std::vector<std::string> filenames = {});




private:

	
	void gsDepthCamera::deepCopyFrameMap(libfreenect2::FrameMap fm_in, libfreenect2::FrameMap &fm_out);

	void gsDepthCamera::waitForFrames();

	bool gsDepthCamera::getNextFrame();

	void gsDepthCamera::popBuffer();

};
