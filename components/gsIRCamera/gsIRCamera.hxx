
#include <deque>

#include <PvSampleUtils.h>
#include <PvDevice.h>
#include <PvDeviceGEV.h>
#include <PvStream.h>
#include <PvStreamGEV.h>
#include <PvPipeline.h>
#include <PvBuffer.h>
#include <PvSystem.h>

#include <opencv2/core.hpp>

class gsIRCamera {

public://variables
	enum State { UNREADY, READY, CLEARING, RECORDING, SAVING, VIRTUAL, DELETED };

private://variables
	PvDeviceGEV *lDeviceGEV;
	PvStreamGEV *lStreamGEV;
	PvPipeline *lPipeline;
	PvString lConnectionID;
	const uint32_t BUFFER_COUNT_MIN = 16;  
	const uint32_t BUFFER_COUNT_MAX = 512; 
	const uint IMAGE_RETRIEVAL_TIMEOUT = 2000;
	const uint64_t DISPLAY_DELAY_LIMIT = 70000; 
	const uint64_t TICK_FREQ = 2083333;
	uint64_t internal_clock_reset, recstart, recend; 
	State state;
	std::string id;
	std::string lFolder;
	std::vector<cv::Mat> virtualBuffer;
	std::vector<cv::Mat>::iterator virtualCurrent;
	std::vector<uint64_t> virtualTimestamps;
	//bool virtual_;

public://methods

	gsIRCamera::gsIRCamera(std::string aConnectionID);
	gsIRCamera::gsIRCamera(std::string pathToFolder, bool camIsVirtual);
	gsIRCamera::~gsIRCamera();

	void gsIRCamera::initialize();

	void gsIRCamera::performNUC();

	void gsIRCamera::clearBuffer();

	void gsIRCamera::startStreaming();

	void gsIRCamera::stopStreaming();

	void gsIRCamera::getImage(cv::Mat &aCvImage, uint64_t timestamp);
	void gsIRCamera::getImage(cv::Mat &aCvImage, uint64_t *timestamp = nullptr);

	void gsIRCamera::startRecording();

	bool gsIRCamera::seek(uint64_t timestamp);

	void gsIRCamera::stopRecordingAndSave(std::string foldername);
	void gsIRCamera::snapshot(std::string foldername, bool jpeg = false, std::string filename = "");

	std::string gsIRCamera::getID();

private://methods

	void gsIRCamera::pv2cv(PvImage *aPvImage, cv::Mat &aCvImage);
	
}; //class gsIR
