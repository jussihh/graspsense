

#include <PvSampleUtils.h>
#include <PvDevice.h>
#include <PvDeviceGEV.h>
#include <PvStream.h>
#include <PvStreamGEV.h>
#include <PvPipeline.h>
#include <PvBuffer.h>
#include <PvSystem.h>

#include <chrono>
#include <stdexcept>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "gsIRCamera.hxx"




double FLIRtoC(double sensor_value) {
	return 0.04*sensor_value - 273.15;
}


gsIRCamera::gsIRCamera(std::string aConnectionID) :
	id(aConnectionID), lDeviceGEV(NULL), lStreamGEV(NULL), lPipeline(NULL), lConnectionID(aConnectionID.c_str()), state(State::UNREADY) {
	
};

gsIRCamera::gsIRCamera(std::string pathToFolder, bool camIsVirtual) :
	lFolder(pathToFolder), state(State::VIRTUAL), virtualBuffer(), virtualTimestamps() {
	cv::FileStorage fs_cam(lFolder + "/camera_data.xml", cv::FileStorage::READ);
	cv::FileStorage fs_frame(lFolder + "/frame_data.xml", cv::FileStorage::READ);

	int n_frames;
	double recstartd, recendd;
	fs_cam["ID"] >> id;
	fs_cam["n_frames"] >> n_frames;
	fs_cam["recstart"] >> recstartd;
	fs_cam["recend"] >> recendd;
	recstart = static_cast<uint64_t>(recstartd);
	recend = static_cast<uint64_t>(recendd);

	std::vector<double> timestamps;
	fs_frame["timestamps"] >> timestamps;
	std::vector<uint64_t> timestampsuint64;
	for (int i = 0; i < timestamps.size(); i++){
		timestampsuint64.push_back(static_cast<uint64_t>(timestamps[i]));
	}
	virtualTimestamps = timestampsuint64;
	fs_cam.release();
	fs_frame.release();

};


gsIRCamera::~gsIRCamera() {
	if (state != State::VIRTUAL) {
		PvGenParameterArray *lDeviceParams = lDeviceGEV->GetParameters();

		delete(lPipeline);
		lStreamGEV->Close();
		PvStream::Free(lStreamGEV);
		lDeviceGEV->Disconnect();
		PvDevice::Free(lDeviceGEV);
	}
};

void gsIRCamera::initialize() {

	if (state != State::VIRTUAL) {
		PvResult lResult;

		lDeviceGEV = dynamic_cast<PvDeviceGEV *>(PvDevice::CreateAndConnect(lConnectionID, &lResult));
		if (!lResult.IsOK()) {
			throw(std::runtime_error("Could not connect to " + static_cast<std::string>(lConnectionID)));
		}

		lStreamGEV = static_cast<PvStreamGEV *>(PvStream::CreateAndOpen(lConnectionID, &lResult));
		if (!lResult.IsOK()) {
			throw(std::runtime_error("Could not open stream to " + static_cast<std::string>(lConnectionID)));
		}

		lDeviceGEV->NegotiatePacketSize();
		lDeviceGEV->SetStreamDestination(lStreamGEV->GetLocalIPAddress(), lStreamGEV->GetLocalPort());

		PvGenParameterArray *lDeviceParams = lDeviceGEV->GetParameters();
		lDeviceParams->SetEnumValue("PixelFormat", "Mono14");
		lDeviceParams->SetEnumValue("CMOSBitDepth", "bit14bit");
		lDeviceParams->SetEnumValue("TemperatureLinearMode", "On");
		lDeviceParams->SetEnumValue("SensorGainMode", "HighGainMode");
		lDeviceParams->SetEnumValue("TemperatureLinearResolution", "High"); // THIS MEANS THAT UNIT IS 0.04K
		lDeviceParams->SetEnumValue("NUCMode", "Manual");

		lPipeline = new PvPipeline(static_cast<PvStream*>(lStreamGEV));
		if (lPipeline != NULL) {
			uint32_t lSize = lDeviceGEV->GetPayloadSize();
			lPipeline->SetBufferCount(BUFFER_COUNT_MAX);
			lPipeline->SetBufferSize(lSize);
		}
		else {
			throw(std::runtime_error("Could not set up a pipeline with " + static_cast<std::string>(lConnectionID)));;
		}
	}
	else {

		for (int i = 0; i < virtualTimestamps.size(); i++) {
			cv::Mat cvimage = cv::imread(lFolder + "/" + getID() + "_" + std::to_string(virtualTimestamps[i]) + ".exr", cv::IMREAD_GRAYSCALE|cv::IMREAD_ANYDEPTH);
			cvimage.convertTo(cvimage, CV_16UC1);
			printf("Loaded image %i in camera %s.\n", i, getID());

			virtualBuffer.push_back(cvimage);
		}
		virtualCurrent = virtualBuffer.begin();

	}
}

void gsIRCamera::performNUC() {
	PvGenParameterArray *lDeviceParams = lDeviceGEV->GetParameters();
	PvGenCommand *NUC = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("NUCAction"));
	std::cout << "Performing NUC on " + getID() << std::endl;
	NUC->Execute();
}

void gsIRCamera::startStreaming() {
	PvGenParameterArray *lDeviceParams = lDeviceGEV->GetParameters();
	
	PvGenCommand *lStart = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStart"));
	
	PvGenCommand *lClockReset = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("GevTimestampControlReset"));

	std::cout << "Starting pipeline " + static_cast<std::string>(lConnectionID) << endl;
	lPipeline->Start();

	std::cout << "Enabling streaming and sending AcquisitionStart command on " + static_cast<std::string>(lConnectionID) << endl;
	lDeviceGEV->StreamEnable();
	lClockReset->Execute();
	internal_clock_reset = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	std::cout << "Internal clock reset: " << internal_clock_reset << std::endl;

	lStart->Execute();
	
	state = State::READY;
}

void gsIRCamera::clearBuffer() {

	
}

void gsIRCamera::stopStreaming() {


	PvGenParameterArray *lDeviceParams = lDeviceGEV->GetParameters();
	lDeviceParams->SetEnumValue("NUCMode", "Automatic");

	PvGenCommand *lStop = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStop"));

	lStop->Execute();
	lDeviceGEV->StreamDisable();
	lPipeline->Stop();
	state = State::UNREADY;
}

void gsIRCamera::startRecording() {
	state = State::RECORDING;
	recstart = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

}

void gsIRCamera::stopRecordingAndSave(std::string foldername) {
	if (state == State::RECORDING) {
		recend = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		cv::FileStorage fs_cam(foldername + "/camera_data.xml", cv::FileStorage::WRITE);
		cv::FileStorage fs_frame(foldername + "/frame_data.xml", cv::FileStorage::WRITE);

		PvGenParameterArray *lDeviceParams = lDeviceGEV->GetParameters();
		PvGenCommand *lStop = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStop"));
		lStop->Execute();
		
		uint32_t nbuffer = lPipeline->GetBufferCount();
		state = State::SAVING;

		std::vector<double> timestamps;

		int nimage = 0;
		while (lPipeline->GetOutputQueueSize() > 0) {
			cv::Mat cvImage, cvImage_float;
			uint64_t timestamp;
			getImage(cvImage, &timestamp);
			if ((timestamp > recstart) && (timestamp < recend)) {
				cvImage.convertTo(cvImage_float, CV_32FC1, 1); //Saved image is in signal value, multiply by 0.04 to get Kelvin
				printf("Saved image %i in camera %s.\n", ++nimage, getID());
				cv::imwrite(foldername + "/" + getID() + "_" + std::to_string(timestamp) + ".exr", cvImage_float);
				timestamps.push_back(static_cast<double>(timestamp));
			}
			else {
				std::cout << "Skipped a buffer outside of recording window." << std::endl;
			}
		}

		fs_cam << "ID" << getID();
		fs_cam << "recstart" << static_cast<double>(recstart);
		fs_cam << "recend" << static_cast<double>(recend);
		fs_cam << "n_frames" << static_cast<int>(timestamps.size());

		fs_frame << "timestamps" << timestamps;

		fs_cam.release();
		fs_frame.release();


		PvGenCommand *lStart = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStart"));
		lStart->Execute();


		state = State::READY;
	}
	
}

void gsIRCamera::snapshot(std::string foldername, bool jpeg, std::string filename) {
	cv::Mat cvImage, cvImage_out;
	uint64_t timestamp;
	getImage(cvImage, &timestamp);
	if (jpeg) {
		double mini, maxi;
		cv::minMaxLoc(cvImage, &mini, &maxi);
		cvImage -= mini;
		cvImage *= (pow(2, 16) / (maxi - mini));
		cvImage.convertTo(cvImage_out, CV_8UC1, 1.0 / 255);
	}
	else{
		cvImage.convertTo(cvImage_out, CV_32FC1, 1);//Saved image is in signal value, multiply by 0.04 to get Kelvin
	}
		

	if (filename == "")
		filename = getID() + "_" + std::to_string(timestamp) + ".exr";


	cv::imwrite(foldername + "/" + filename, cvImage_out);
}

void gsIRCamera::getImage(cv::Mat &aCvImage, uint64_t timestamp) {
	uint64_t ts = 0;
	while((timestamp-16666)>ts)
		getImage(aCvImage, &ts);
}

void gsIRCamera::getImage(cv::Mat &aCvImage, uint64_t *timestamp) {

	if (state != State::VIRTUAL) {
		PvBuffer *lBuffer = NULL;
		PvResult lOperationResult;

		bool gotImage = false;

		while (!gotImage && state!=DELETED) {

			PvResult lResult = lPipeline->RetrieveNextBuffer(&lBuffer, IMAGE_RETRIEVAL_TIMEOUT, &lOperationResult);


			if (lResult.IsOK())
			{
				if (lOperationResult.IsOK())
				{
					if (state == State::READY) {
						uint64_t timenow = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
						while ((timenow - (internal_clock_reset + 1e6 * lBuffer->GetTimestamp() / TICK_FREQ)) > DISPLAY_DELAY_LIMIT) {

							lPipeline->ReleaseBuffer(lBuffer);
							lBuffer = NULL;
							lResult = lPipeline->RetrieveNextBuffer(&lBuffer, IMAGE_RETRIEVAL_TIMEOUT, &lOperationResult);
						}
					}

					PvPayloadType lType;

					uint32_t lWidth = 0, lHeight = 0;
					lType = lBuffer->GetPayloadType();

					if (lType == PvPayloadTypeImage)
					{
						gotImage = true;
						PvImage *lImage = lBuffer->GetImage();

						gsIRCamera::pv2cv(lImage, aCvImage);
						if (timestamp != nullptr)
							*timestamp = static_cast<uint64_t>(internal_clock_reset + 1e6 * lBuffer->GetTimestamp() / TICK_FREQ);

					}
					else {
						std::cout << " (buffer does not contain image)\n";
					}
				}
				else
				{
					std::cout << "PvResult Code: " << lOperationResult.GetCode() << std::endl;
					std::cout << "PvResult Desc: " << lOperationResult.GetDescription() << std::endl;
				}

				if(gotImage)
					lPipeline->ReleaseBuffer(lBuffer);


			}
			else
			{
				std::cout << std::endl << "Buffer empty!" << std::endl;

			}
		}
	}
	else {
		aCvImage = *virtualCurrent;
		virtualCurrent++;
		if (virtualCurrent == virtualBuffer.end())
			virtualCurrent = virtualBuffer.begin();
		if (timestamp != nullptr)
			*timestamp = virtualTimestamps[std::distance(virtualBuffer.begin(), virtualCurrent)];
	}
}

bool gsIRCamera::seek(uint64_t timestamp) {
	if (state == State::VIRTUAL) {

		auto lb = std::lower_bound(virtualTimestamps.begin(), virtualTimestamps.end(), timestamp);

		if (lb == virtualTimestamps.end()) {
			printf("Seek result: timepoint outside of recording: %lld\n", timestamp);
			return false;
		}

		if (lb == virtualTimestamps.begin()) {
			printf("Seek result: timepoint before first frame, returning first frame, diff: %lld\n", *lb - timestamp);
			virtualCurrent = virtualBuffer.begin();
			return true;
		}

		ptrdiff_t index;
		if ((*lb - timestamp) < (timestamp - *(lb - 1))) {
			index = std::distance(virtualTimestamps.begin(), lb);
		}
		else {
			index = std::distance(virtualTimestamps.begin(), (lb - 1));
		}

		virtualCurrent = virtualBuffer.begin() + index;

		return true;

	}
	else {
		return false;
	}
}

void gsIRCamera::pv2cv(PvImage *aPvImage, cv::Mat &aCvImage) {
	uint8_t *pvDataPtr = aPvImage->GetDataPointer();
	aCvImage = cv::Mat(aPvImage->GetHeight(), aPvImage->GetWidth(), CV_16UC1, pvDataPtr, cv::Mat::AUTO_STEP);
}

std::string gsIRCamera::getID() {
	return id;
}

