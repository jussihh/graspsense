

#pragma once

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <opencv2/core.hpp> 
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> 

#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <functional>
#include <stdexcept>

#include "gsDepthCamera.hxx"
#include "gsDepthCamera_serialization.hxx"
#include "gsTimer.hxx"


//live kinect
gsDepthCamera::gsDepthCamera(gsDepthCamera::Processor p, std::string serial) :
	listener_(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth),
	undistorted_(512, 424, 4),
	registered_(512, 424, 4),
	big_mat_(1920, 1082, 4),
	serial_(serial),
	state_(State::UNREADY),
	frameBuffer_(),
	bufferHasData_(false),
	bufferHasDataMutex_(),
	frameMutex_(),
	frameThread_(&gsDepthCamera::waitForFrames, this),
	previousFrameStamp_(0)
{


	if (freenect2_.enumerateDevices() == 0)
	{
		std::cout << "no kinect2 connected!" << std::endl;
		exit(-1);
	}

	switch (p)
	{
	case OPENGL:
		std::cout << "Creating a OpenGL processor." << std::endl;
		if (serial.empty())
			dev_ = freenect2_.openDefaultDevice(new libfreenect2::OpenGLPacketPipeline());
		else
			dev_ = freenect2_.openDevice(serial, new libfreenect2::OpenGLPacketPipeline());
		break;
#ifdef WITH_CUDA
	case CUDA:
		std::cout << "Creating a CUDA processor." << std::endl;
		if (serial.empty())
			dev_ = freenect2_.openDefaultDevice(new libfreenect2::CudaPacketPipeline());
		else
			dev_ = freenect2_.openDevice(serial, new libfreenect2::CudaPacketPipeline());
		break;
	case CUDADUMP:
		std::cout << "Creating a CUDADUMP processor." << std::endl;
		if (serial.empty())
			dev_ = freenect2_.openDefaultDevice(new libfreenect2::CudaDumpPacketPipeline());
		else
			dev_ = freenect2_.openDevice(serial, new libfreenect2::CudaDumpPacketPipeline());
		break;
#endif
	default:
		std::cout << "Creating a CPU processor." << std::endl;
		if (serial.empty())
			dev_ = freenect2_.openDefaultDevice(new libfreenect2::CpuPacketPipeline());
		else
			dev_ = freenect2_.openDevice(serial, new libfreenect2::CpuPacketPipeline());
		break;
	}

	serial_ = dev_->getSerialNumber();
	dev_->setColorFrameListener(&listener_);
	dev_->setIrAndDepthFrameListener(&listener_);
	
	if (!dev_->start())
		throw std::runtime_error("Cannot start Kinect device.");

	registration_ = new libfreenect2::Registration(dev_->getIrCameraParams(), dev_->getColorCameraParams());

	state_ = State::READY;

}


gsDepthCamera::gsDepthCamera(std::string path_to_folder) :
	listener_(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth),
	undistorted_(512, 424, 4),
	registered_(512, 424, 4),
	big_mat_(1920, 1082, 4),
	//serial_(serial),
	state_(State::UNREADY),
	frameBuffer_(),
	bufferHasData_(false),
	bufferHasDataMutex_(),
	frameMutex_(),
	path_to_folder_(path_to_folder),
	previousFrameStamp_(0)
{
	
	libfreenect2::Freenect2Device::IrCameraParams irparams;
	libfreenect2::Freenect2Device::ColorCameraParams colorparams;


	std::ifstream cam_data_stream(path_to_folder_ + "/camera_data.txt", std::ifstream::in);
	boost::archive::text_iarchive cam_data_archive(cam_data_stream);

	cam_data_archive >> serial_;
	cam_data_archive >> colorparams;
	cam_data_archive >> irparams;
	cam_data_archive >> n_frames_;

	registration_ = new libfreenect2::Registration(irparams, colorparams);

}

gsDepthCamera::~gsDepthCamera() {

	if(frameThread_.joinable())
		frameThread_.join();

	delete(registration_);
}

bool gsDepthCamera::loadData() {
	std::ifstream frame_data_stream(path_to_folder_ + "/frame_data.txt");
	boost::archive::text_iarchive frame_data_archive(frame_data_stream);

	for (int i = 0; i<n_frames_; i++) {
		printf("Loading frame %i in camera %s.\n", i, getSerial().c_str());
		libfreenect2::FrameMap tempmap, tempmap2;
		frame_data_archive >> tempmap;
		cv::Mat ir = cv::imread(path_to_folder_ + "/" + getSerial() + "_ir_" + std::to_string(tempmap[libfreenect2::Frame::Ir]->received_timestamp) + ".exr", cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH) * 65535;
		cv::Mat depth = cv::imread(path_to_folder_ + "/" + getSerial() + "_depth_" + std::to_string(tempmap[libfreenect2::Frame::Depth]->received_timestamp) + ".exr", cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH) * 10000;
		cv::Mat color3C = cv::imread(path_to_folder_ + "/" + getSerial() + "_color_" + std::to_string(tempmap[libfreenect2::Frame::Color]->received_timestamp) + ".jpg");
		
		cv::Mat color4C = cv::Mat(color3C.rows, color3C.cols, CV_8UC4);
		int fromto[] = { 0,0, 1,1, 2,2, 3,3 };
		cv::mixChannels({ { color3C } }, { { color4C } }, fromto, color3C.channels());

		tempmap[libfreenect2::Frame::Ir]->data = reinterpret_cast<unsigned char *>(ir.data);
		tempmap[libfreenect2::Frame::Depth]->data = reinterpret_cast<unsigned char *>(depth.data);
		tempmap[libfreenect2::Frame::Color]->data = reinterpret_cast<unsigned char *>(color4C.data);

		deepCopyFrameMap(tempmap, tempmap2);
		frameBuffer_.push_back(tempmap2);

	}
	
	virtualCamIterator_ = frameBuffer_.begin();

	state_ = gsDepthCamera::VIRTUAL;
	return true;
}
std::vector<std::string> gsDepthCamera::getSerials() {
	int n_kinect = freenect2_.enumerateDevices();
	std::vector<std::string> serials;
	for (int i = 0; i < n_kinect; i++) {
		serials.push_back(freenect2_.getDeviceSerialNumber(i));
	}
	return serials;
}

std::string gsDepthCamera::getSerial() {
	return serial_;
}

gsDepthCamera::State gsDepthCamera::state() {
	return state_;
}

void gsDepthCamera::shutDown() {
	if (state_ != State::VIRTUAL) {
		state_ = State::DELETED;
		while(!listener_.hasNewFrame());
		dev_->stop();
		dev_->close();
	}
	else {
		state_ = State::DELETED;
	}
}

void gsDepthCamera::clearBuffer() {

	State initial_state = state_;
	state_ = State::CLEARING;
	while (!frameBuffer_.empty()) {
		popBuffer();
	}
	state_ = initial_state;
}

bool gsDepthCamera::getImages(cv::Mat &ir_mat, cv::Mat &depth_mat, cv::Mat &color_mat, bool registered, std::vector<uint64_t> *stamps) {
	std::lock_guard<std::mutex> lock(getImagesMutex_);
	if (!gsDepthCamera::getNextFrame())
		return false;

	libfreenect2::Frame *ir = frames_[libfreenect2::Frame::Ir];
	libfreenect2::Frame *depth = frames_[libfreenect2::Frame::Depth];
	libfreenect2::Frame *color = frames_[libfreenect2::Frame::Color];
	if (stamps != nullptr) {
		stamps->push_back(ir->received_timestamp);
		stamps->push_back(depth->received_timestamp);
		stamps->push_back(color->received_timestamp);
	}

	cv::Mat tmp_ir;
	cv::Mat tmp_depth;
	cv::Mat tmp_color;

	if (registered) {

	}
	else {
		tmp_ir = cv::Mat(static_cast<int>(ir->height), static_cast<int>(ir->width), CV_32FC1, ir->data);
		tmp_depth = cv::Mat(static_cast<int>(depth->height), static_cast<int>(depth->width), CV_32FC1, depth->data);
		tmp_color = cv::Mat(static_cast<int>(color->height), static_cast<int>(color->width), CV_8UC4, color->data);
	}
	
	if (state_ != State::VIRTUAL) {
		cv::flip(tmp_depth, depth_mat, 1);
		cv::flip(tmp_color, color_mat, 1);
		cv::flip(tmp_ir, ir_mat, 1);

		depth_mat = depth_mat / 10000.0;
		ir_mat = ir_mat / 65535.0;
	}
	else {
		depth_mat = tmp_depth / 10000.0;
		color_mat = tmp_color;
		ir_mat = tmp_ir / 65535.0;
	}


	if (state_ != State::VIRTUAL) {
		if (state_ != State::SAVING)
			listener_.release(frames_);

		if (state_ != State::RECORDING)
			popBuffer();
	}
	return true;

}

bool gsDepthCamera::seek(uint64_t time_in_microseconds) {
	if (state_ == State::VIRTUAL) {

		std::vector<uint64_t> timestamps;
		for (std::deque<libfreenect2::FrameMap>::iterator it = frameBuffer_.begin(); it != frameBuffer_.end(); ++it) {
			timestamps.push_back((*it)[libfreenect2::Frame::Color]->received_timestamp);
		}
		auto lb = std::lower_bound(timestamps.begin(), timestamps.end(), time_in_microseconds);

		if (lb == timestamps.end()) {
			printf("Seek result: timepoint outside of recording\n");
			return false;
		}
		
		
		if (lb == timestamps.begin()) {
			virtualCamIterator_ = frameBuffer_.begin();
			return true;
		}
			
		ptrdiff_t index;
		if ((*lb - time_in_microseconds) < (time_in_microseconds - *(lb-1))) {
			index = std::distance(timestamps.begin(), lb);
		}
		else {
			index = std::distance(timestamps.begin(), (lb-1));
		}

		virtualCamIterator_ = frameBuffer_.begin() + index;

		return true;
	}
	else {
		return false;
	}
}

bool gsDepthCamera::getNextFrame() {

	if (state_ == State::VIRTUAL) {
		if (!frameBuffer_.empty()) {

			frames_ = *virtualCamIterator_;
			virtualCamIterator_++;
			if (virtualCamIterator_ == frameBuffer_.end())
				virtualCamIterator_ = frameBuffer_.begin();
			
			return true;
		}
		else {
			return false;
		}
	}
	else {
		{
			std::unique_lock<std::mutex> lk(bufferHasDataMutex_);
			while ((state_ == State::READY) && (bufferHasDataCV_.wait_for(lk, std::chrono::milliseconds(30), [this] {return bufferHasData_; }) == false)) {
				
			};
		}
		switch (state_) {
		case State::DELETED:
		case State::RECORDING:
			return false;
		case State::READY:
			{
				std::lock_guard<std::mutex> lock(frameMutex_);
				deepCopyFrameMap(frameBuffer_.front(), frames_);
			}
			break;
		case State::SAVING:
			frames_ = frameBuffer_.front();
			break;
		}
		return true;
	}
}


void gsDepthCamera::deepCopyFrameMap(libfreenect2::FrameMap fm_in, libfreenect2::FrameMap &fm_out) {

	for (libfreenect2::FrameMap::iterator it = fm_in.begin(); it != fm_in.end(); ++it)
	{
		size_t sizeofdata = it->second->width * it->second->height * it->second->bytes_per_pixel;
		fm_out[it->first] = new libfreenect2::Frame(it->second->width, it->second->height, it->second->bytes_per_pixel);

		std::memcpy(fm_out[it->first]->data, it->second->data, sizeofdata);
	
		fm_out[it->first]->sequence = it->second->sequence;
		fm_out[it->first]->timestamp = it->second->timestamp; 
		fm_out[it->first]->received_timestamp = it->second->received_timestamp;
		fm_out[it->first]->exposure = it->second->exposure;
		fm_out[it->first]->gain = it->second->gain;
		fm_out[it->first]->gamma = it->second->gamma;
		fm_out[it->first]->format = it->second->format;
		
	}
	
}

void gsDepthCamera::waitForFrames() {
	gsTimer timer1("DeepCopy timer", false);

	while (state_ != State::DELETED) {
		if (state_ != State::SAVING) {

			timer1.checkpoint("Started waiting");
			listener_.waitForNewFrame(new_frames_);
			timer1.checkpoint("Got frames, deepcopying");
			libfreenect2::FrameMap tempFrames;
			deepCopyFrameMap(new_frames_, tempFrames);
			timer1.checkpoint("Done deepcopying");


			uint64_t timediff, received;

			received = new_frames_.begin()->second->received_timestamp;

			if (previousFrameStamp_ < received)
				timediff = received - previousFrameStamp_;
			else
				timediff = previousFrameStamp_ - received;

			previousFrameStamp_ = received;

			if (timediff > 60000)
				std::cout << "Time difference: " << timediff << std::endl;


			listener_.release(new_frames_); 						

			if (frameBuffer_.size() == MAX_BUFFER_SIZE_) {
				popBuffer();
				printf("Buffer full.\n");
			}

			if (state_ != State::CLEARING) {
				{
					std::lock_guard<std::mutex> lock(frameMutex_);
					frameBuffer_.push_back(tempFrames);
				}

				if (frameBuffer_.size() > 0) {
					std::lock_guard<std::mutex> lk(bufferHasDataMutex_);
					bufferHasData_ = true;
				}

				bufferHasDataCV_.notify_one();
			}
			else {
				listener_.release(tempFrames);
			}
		}
	}
}

void gsDepthCamera::popBuffer() {
	std::lock_guard<std::mutex> lock(frameMutex_);
	if (frameBuffer_.size() > 0) {
		for (libfreenect2::FrameMap::iterator it = frameBuffer_.front().begin(); it != frameBuffer_.front().end(); ++it)
		{
			delete it->second;
			it->second = 0;
		}

		if (frameBuffer_.size() == 1) {
			std::lock_guard<std::mutex> lk(bufferHasDataMutex_);
			bufferHasData_ = false;
		}

		frameBuffer_.pop_front();

	}
}

void gsDepthCamera::pause() {
	dev_->stop();
}
void gsDepthCamera::unpause() {
	dev_->start();
}

void gsDepthCamera::startRecording() {
	
	if (state_ == State::READY) {
		state_ = State::RECORDING;	
		recStartTime_ = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	}
	else {
		throw(std::runtime_error("Cannot start recording\n"));
	}
}

void gsDepthCamera::stopRecordingAndSave(std::string foldername) {
	if (state_ == State::RECORDING) {
		recStopTime_ = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

		state_ = State::SAVING;

		std::ofstream cam_data_stream(foldername + "/camera_data.txt");
		std::ofstream frame_data_stream(foldername + "/frame_data.txt");

		boost::archive::text_oarchive cam_data_archive(cam_data_stream);
		boost::archive::text_oarchive frame_data_archive(frame_data_stream);

		cv::Mat ir, depth, color;
		std::string irfn, depthfn, colorfn;
		std::string fnprefix = foldername + "/" + getSerial() + "_";

		int n_frames = 0;

		while (!frameBuffer_.empty()) {
			bool insideRecWindow = false;
			{
				std::lock_guard<std::mutex> lock(frameMutex_);
				libfreenect2::FrameMap tempmap = frameBuffer_.front();

				if (tempmap[libfreenect2::Frame::Color]->received_timestamp > recStartTime_ && tempmap[libfreenect2::Frame::Color]->received_timestamp < recStopTime_)
					insideRecWindow = true;

				if (insideRecWindow) {
					frame_data_archive << tempmap; 
					irfn = fnprefix + "ir_" + std::to_string(tempmap[libfreenect2::Frame::Ir]->received_timestamp) + ".exr";
					depthfn = fnprefix + "depth_" + std::to_string(tempmap[libfreenect2::Frame::Depth]->received_timestamp) + ".exr";
					colorfn = fnprefix + "color_" + std::to_string(tempmap[libfreenect2::Frame::Color]->received_timestamp) + ".jpg";
				}
			}
			if (insideRecWindow) {
				getImages(ir, depth, color, false);
				cv::imwrite(irfn, ir);
				cv::imwrite(depthfn, depth);
				cv::imwrite(colorfn, color); 
				n_frames++;
				printf("Saved frame %i in camera %s.\n", n_frames, getSerial().c_str());
			}
			else {
				printf("Discarded frame outside of the recording window.\n");
				popBuffer();
			}
			
		}

		cam_data_archive << getSerial(); 
		cam_data_archive << dev_->getColorCameraParams();
		cam_data_archive << dev_->getIrCameraParams();
		cam_data_archive << n_frames;

		state_ = State::READY;
	}
}

void gsDepthCamera::snapshot(std::string foldername, bool jpeg, std::vector<std::string> filenames) {

	cv::Mat ir, depth, color, ir8bit, depth8bit;

	std::vector<uint64_t> stamps;

	getImages(ir, depth, color, false, &stamps);
	std::string fnprefix = foldername + "/";

	std::vector<std::string> defaultfilenames = {
		getSerial() + "_ir_"    + std::to_string(stamps[0]) + (jpeg ? ".jpg" : ".exr"),
		getSerial() + "_depth_" + std::to_string(stamps[1]) + ".exr",
		getSerial() + "_color_" + std::to_string(stamps[2]) + ".jpg"
	};

	if (filenames.empty()) {
		std::cout << "Using default filenames.\n";
		filenames = defaultfilenames;
	}
	
	if (jpeg) {
		ir.convertTo(ir8bit, CV_8UC1, 255.0);
		ir = ir8bit;
	}

	if (filenames[0] != "") cv::imwrite(fnprefix + filenames[0], ir);
	if (filenames[1] != "") cv::imwrite(fnprefix + filenames[1], depth);
	if (filenames[2] != "") cv::imwrite(fnprefix + filenames[2], color);

}
