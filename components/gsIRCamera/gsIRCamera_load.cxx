
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
	//load frames
	cam->initialize();
}

void getImagesFromCamera(unique_ptr<gsIRCamera> &cam, cv::Mat &image) {
	cv::Mat lCvImage, tempCvImage;
	cv::Mat falseColorImage;
	cam->getImage(lCvImage);
	tempCvImage = lCvImage.clone();	
	double mini, maxi;
	cv::minMaxLoc(tempCvImage, &mini, &maxi);
	tempCvImage -= mini;
	tempCvImage *= (65536.0 / (maxi - mini));
	tempCvImage.convertTo(falseColorImage, CV_8UC3, 1.0 / 255);
	cv::applyColorMap(falseColorImage, falseColorImage, cv::COLORMAP_JET);
	image = falseColorImage;
}


int main() {

	std::vector<std::unique_ptr<gsIRCamera>> lgsIRCameras;
	lgsIRCameras.push_back(std::make_unique<gsIRCamera>("./experiment1/192.168.1.101", true));
	lgsIRCameras.push_back(std::make_unique<gsIRCamera>("./experiment1/192.168.1.102", true));
	lgsIRCameras.push_back(std::make_unique<gsIRCamera>("./experiment1/192.168.1.103", true));
	lgsIRCameras.push_back(std::make_unique<gsIRCamera>("./experiment1/192.168.1.104", true));

	cv::FileStorage fs("./experiment1/experiment.xml", cv::FileStorage::READ);
	double recend, recstart;
	fs["recstart"] >> recstart;
	fs["recend"] >> recend;
	fs.release();

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
		gsTimer displayTimer("Display timer");
		uint64_t current_time = recstart;
		while(true)
		{
			for (int i = 0; i < images.size(); i++) {
				lgsIRCameras[i]->seek(current_time);
				getThreads.push_back(std::thread(getImagesFromCamera, std::ref(lgsIRCameras[i]), std::ref(images[i])));
			}
			for (int i = 0; i < getThreads.size(); i++) {
				if (getThreads[i].joinable())
					getThreads[i].join();
			}
			for (int i = 0; i < images.size(); i++) {
				imshow(lgsIRCameras[i]->getID(), images[i]);
			}
			int waitfor = 1000.0 / 30 - displayTimer.getElapsed();
			int c = cv::waitKey((waitfor>1) ? waitfor : 1);
			displayTimer.checkpoint("Displayed");
			current_time += 1e6 / 30.0;
			if (current_time > recend)
				current_time = recstart;
			if (c != -1)
				break;
		}
	}
	catch (std::runtime_error &aexception) {
		cout << aexception.what() << std::endl;
		throw;
	}
}
