
#include <stdexcept>
#include <memory>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "gsIRCamera.hxx"
#include "gsTimer.hxx"

#include <boost/filesystem.hpp>

void startCamera(unique_ptr<gsIRCamera> &cam) {
	cam->initialize();
	cam->startStreaming();
}
void stopRecording(unique_ptr<gsIRCamera> &cam) {
	std::string foldername = "./experiment1/" + cam->getID();
	boost::filesystem::create_directory(foldername);
	cam->stopRecordingAndSave(foldername);
}

void getImagesFromCamera(unique_ptr<gsIRCamera> &cam, cv::Mat &image) {
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

int main() {

	std::vector<std::unique_ptr<gsIRCamera>> lgsIRCameras;
	lgsIRCameras.push_back(std::make_unique<gsIRCamera>("192.168.1.101"));
	lgsIRCameras.push_back(std::make_unique<gsIRCamera>("192.168.1.102"));
	lgsIRCameras.push_back(std::make_unique<gsIRCamera>("192.168.1.103"));
	lgsIRCameras.push_back(std::make_unique<gsIRCamera>("192.168.1.104"));

	try {
		std::vector<std::thread> startThreads;
		for (unique_ptr<gsIRCamera> &lgsIRCamera : lgsIRCameras) {
			startThreads.push_back(std::thread(startCamera, std::ref(lgsIRCamera)));
			cv::namedWindow(lgsIRCamera->getID());
		}

		for (int i = 0; i < startThreads.size(); i++) {
			startThreads[i].join();
		}

		std::vector<cv::Mat> images(lgsIRCameras.size());
		std::vector<std::thread> getThreads;
		gsTimer recStart("Recording");
		gsTimer recEnd("Recording");
		bool recordingstarted = false;
		bool saved = false;
		uint64_t recstart, recend;
		while(true)
		{
			if (!recordingstarted && recStart.getElapsed() > 5000) {
				for (unique_ptr<gsIRCamera> &lgsIRCamera : lgsIRCameras) {
					lgsIRCamera->startRecording();
				}
				recstart = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
				recordingstarted = true;
			}


			if (recordingstarted && recEnd.getElapsed() > 5500) {
				boost::filesystem::create_directory("./experiment1");
				std::vector<std::thread> stopRecThreads;
				for (unique_ptr<gsIRCamera> &lgsIRCamera : lgsIRCameras) {
					stopRecThreads.push_back(std::thread(stopRecording, std::ref(lgsIRCamera)));
				}
				recend = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

				for (int i = 0; i < stopRecThreads.size(); i++) {
					if(stopRecThreads[i].joinable())
						stopRecThreads[i].join();
				}
				saved = true;
				cv::FileStorage fs("./experiment1/experiment.xml", cv::FileStorage::WRITE);
				std::vector<std::string> cameraids;
				for (unique_ptr<gsIRCamera> &lgsIRCamera : lgsIRCameras) {
					cameraids.push_back(lgsIRCamera->getID());
					lgsIRCamera->performNUC();
				}
				fs << "cameraids" << cameraids;
				fs << "recstart" << static_cast<double>(recstart);
				fs << "recend" << static_cast<double>(recend);
				fs.release();
				break;
			}

			//DISPLAY
			if (!recordingstarted) {
				for (int i = 0; i < images.size(); i++) {
					getThreads.push_back(std::thread(getImagesFromCamera, std::ref(lgsIRCameras[i]), std::ref(images[i])));
				}
				for (int i = 0; i < getThreads.size(); i++) {
					if (getThreads[i].joinable())
						getThreads[i].join();
				}
				for (int i = 0; i < images.size(); i++) {
					imshow(lgsIRCameras[i]->getID(), images[i]);
				}
			}

			//KBCHECK
			if (cv::waitKey(1) != -1)
				break;
		}
		for (unique_ptr<gsIRCamera> &lgsIRCamera : lgsIRCameras) {
			lgsIRCamera->stopStreaming();
		}		
	}
	catch (std::runtime_error &aexception) {
		cout << aexception.what() << std::endl;
		throw;
	}
}
