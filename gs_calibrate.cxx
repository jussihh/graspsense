
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/point_cloud_handlers.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/ccalib/omnidir.hpp"
#include "opencv2/ccalib/multicalib.hpp"
#include "opencv2/ccalib/randpattern.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <string>
#include <thread>      
#include <list>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


#define PI 3.14159265

float incident_angle(Eigen::Vector3f point, Eigen::Matrix3f rotation) {
	Eigen::Vector3f u(-point), v(0.0, 0.0, 1.0);
	v = rotation*v;

	float dot = u.dot(v);
	return acos( dot/(u.norm()*v.norm()) );
}


float interpolate_z(cv::Vec2f point, cv::Mat &depthimage) {

	float x = point[0];
	float y = point[1];

	int x_f = static_cast<int>(std::floor(x));
	int x_c = static_cast<int>(std::ceil(x));
	int y_f = static_cast<int>(std::floor(y));
	int y_c = static_cast<int>(std::ceil(y));

	int w, h;
	w = depthimage.cols;
	h = depthimage.rows;

	if (x_f < 0 || y_f < 0 || x_c >= w || y_c >= h )
		return 0;

	float ff_depth, fc_depth, cf_depth, cc_depth;
	ff_depth = depthimage.at<float>(y_f, x_f);
	fc_depth = depthimage.at<float>(y_f, x_c);
	cf_depth = depthimage.at<float>(y_c, x_f);
	cc_depth = depthimage.at<float>(y_c, x_c);

	if (ff_depth < 500 || fc_depth < 500 || cf_depth < 500 || cc_depth < 500)
		return 0;

	if (ff_depth > 1400 || fc_depth > 1400 || cf_depth > 1400 || cc_depth > 1400)
		return 0;

	std::vector<float> depths{ ff_depth, fc_depth, cf_depth, cc_depth };
	auto min_and_max =  std::minmax(depths.begin(), depths.end());

	if (std::abs(min_and_max.first - min_and_max.second) > 20)
		return 0;


	float ff_dist = std::sqrt(std::pow(x - x_f, 2) + std::pow(y - y_f, 2));
	float fc_dist = std::sqrt(std::pow(x - x_f, 2) + std::pow(y - y_c, 2));
	float cf_dist = std::sqrt(std::pow(x - x_c, 2) + std::pow(y - y_f, 2));
	float cc_dist = std::sqrt(std::pow(x - x_c, 2) + std::pow(y - y_c, 2));

	float total_dist = ff_dist + fc_dist + cf_dist + cc_dist;

	float depth = ff_dist / total_dist * ff_depth + fc_dist / total_dist * fc_depth + cf_dist / total_dist * cf_depth + cc_dist / total_dist * cc_depth;

	return depth;
}

cv::Mat gsimread(std::string filepath) {
	std::string extension = filepath.substr(filepath.rfind('.'), filepath.size()-1);
	cv::Mat outputimage;
	if (extension == ".exr") {
		double scale = 1.0;
		double minK, maxK;
		cv::Mat inputimage_raw = cv::imread(filepath, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH)*scale;
		if (inputimage_raw.cols > 1000) {
			minK = 0.0;
			maxK = 200.0;
		}
		else {
			minK = 273.15 + 30;
			maxK = 273.15 + 45;
		}

		inputimage_raw = (inputimage_raw - minK) / (maxK - minK);
		inputimage_raw.convertTo(outputimage, CV_8UC1, 255);
		if (outputimage.cols > 1000) 
			outputimage = 255 - outputimage;
	}
	else {
		outputimage = 255-cv::imread(filepath, 0);
	}
	imshow("OUTPUT", outputimage);

	return outputimage;
}


static void StereoCalib(const std::vector<std::string>& imagelist, cv::Size boardSize, float squareSize, cv::Mat &cameraMatrix1, cv::Mat &distCoeffs1, cv::Mat &cameraMatrix2, cv::Mat &distCoeffs2, cv::Mat &R, cv::Mat &T, bool displayCorners = true)
{

	if (imagelist.size() % 2 != 0)
	{
		cout << "Error: the image list contains odd (non-even) number of elements\n";
		return;
	}
	std::cout << imagelist[0];
	const int maxScale = 2;

	std::vector<std::vector<cv::Point2f> > imagePoints[2];
	std::vector<std::vector<cv::Point3f> > objectPoints;
	std::vector<cv::Size> imageSize(2);

	int i, j, k, nimages = (int)imagelist.size() / 2;

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	std::vector<std::string> goodImageList;

	for (i = j = 0; i < nimages; i++)
	{
		for (k = 0; k < 2; k++)
		{
			const std::string& filename = imagelist[i * 2 + k];

			cv::Mat img = gsimread(filename);
			if (img.empty())
				break;
			if (imageSize[1 - i%2] == cv::Size())
				imageSize[1 - i%2] = img.size();
			bool found = false;

			std::vector<cv::Point2f>& centers = imagePoints[k][j];
			for (int scale = 1; scale <= maxScale; scale++)
			{
				cv::Mat timg;
				if (scale == 1)
					timg = img;
				else
					cv::resize(img, timg, cv::Size(), scale, scale);

				cv::SimpleBlobDetector::Params cDp = cv::SimpleBlobDetector::Params();
				cDp.thresholdStep = 3;
				cDp.minThreshold = 15;
				cDp.maxThreshold = 245;
				cDp.minRepeatability = 3;

				cv::Ptr<cv::FeatureDetector> circleDetector = cv::SimpleBlobDetector::create(cDp);

				found = cv::findCirclesGrid(timg, boardSize, centers, cv::CALIB_CB_ASYMMETRIC_GRID | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_CLUSTERING, circleDetector);

				if (found)
				{
					if (scale > 1)
					{
						cv::Mat centersMat(centers);
						centersMat *= 1. / scale;

					}
					break;
				}
			}
			if (displayCorners)
			{
				cout << filename << endl;
				cv::Mat cimg, cimg1;
				cvtColor(img, cimg, cv::COLOR_GRAY2BGR);
				drawChessboardCorners(cimg, boardSize, centers, found);
				double sf = 640. / MAX(img.rows, img.cols);
				resize(cimg, cimg1, cv::Size(), sf, sf);
				imshow("corners", cimg1);
				cv::imwrite(filename + "_points.jpg", cimg);
				char c = (char)cv::waitKey(500);
				if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
					exit(-1);
			}
			else
				putchar('.');
			if (!found)
				break;

		}
		if (k == 2)
		{
			goodImageList.push_back(imagelist[i * 2]);
			goodImageList.push_back(imagelist[i * 2 + 1]);
			j++;
		}
	}
	cout << j << " pairs have been successfully detected.\n";
	nimages = j;
	if (nimages < 2)
	{
		cout << "Error: too few pairs to run the calibration\n";
		return;
	}
	
	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	objectPoints.resize(nimages);

	for (i = 0; i < nimages; i++)
	{
		for (int j = 0; j < boardSize.height; j++)
			for (int k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(cv::Point3f(float((2 * k + j % 2)*squareSize), float(j*squareSize), 0));
	}


	cout << "Running stereo calibration ...\n";

	cv::Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = cameraMatrix1;
	cameraMatrix[1] = cameraMatrix2;
	distCoeffs[0] = distCoeffs1;
	distCoeffs[1] = distCoeffs2;
	cv::Mat E, F;

	double rms = cv::stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize[0], R, T, E, F, cv::CALIB_FIX_INTRINSIC, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));

	cout << "done with RMS error=" << rms << endl;
	std::cout << "R:" << std::endl << R << std::endl;
	std::cout << "T:" << std::endl << T << std::endl;
	double err = 0;
	int npoints = 0;
	std::vector<cv::Vec3f> lines[2];
	for (i = 0; i < nimages; i++)
	{
		int npt = (int)imagePoints[0][i].size();
		cv::Mat imgpt[2];
		for (k = 0; k < 2; k++)
		{
			imgpt[k] = cv::Mat(imagePoints[k][i]);
			undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], cv::Mat(), cameraMatrix[k]);
			computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
		}
		for (j = 0; j < npt; j++)
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] +
					imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	cout << "average epipolar err = " << err / npoints << endl;


}


static double computeReprojectionErrors(const std::vector<std::vector<cv::Point3f> >& objectPoints,
	const std::vector<std::vector<cv::Point2f> >& imagePoints,
	const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
	const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
	std::vector<float>& perViewErrors, bool fisheye)
{
	std::vector<cv::Point2f> imagePoints2;
	size_t totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());
	for (size_t i = 0; i < objectPoints.size(); ++i)
	{
		projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
		err = norm(imagePoints[i], imagePoints2, cv::NORM_L2);
size_t n = objectPoints[i].size();
perViewErrors[i] = (float)std::sqrt(err*err / n);
totalErr += err*err;
totalPoints += n;
	}
	return std::sqrt(totalErr / totalPoints);
}

static void sdmean(std::vector<float> data, double &sd, double &mean) {

	double sum(0), variance(0);
	int n = (int)data.size();

	if (n < 2) {
		if (n == 1) {
			mean = data[0];
			sd = 0.0;
		}
		return;
	}

	for (float xi : data) {
		sum += xi;
	}

	mean = sum / n;

	for (float xi : data) {
		variance += std::pow(xi-mean,2)/(n-1);
	}

	sd = std::sqrt(variance);
}

static void InternalCalib(std::vector<std::string> imagelist, cv::Size boardSize, float squareSize, cv::Mat &cameraMatrix, cv::Mat &distCoeffs, bool displayCorners = true) {

	const int maxScale = 2;


	std::vector<std::vector<cv::Point2f>> imagePoints;
	std::vector<std::vector<cv::Point3f>> objectPoints;
	cv::Size imageSize;

	int i, i_valid;
	int nimages = (int)imagelist.size();
	
	imagePoints.resize(nimages);

	for (i = i_valid = 0; i < nimages; i++)
	{
		std::cout << imagelist[i];
		const std::string& filename = imagelist[i];
		cv::Mat img = gsimread(filename);
		if (img.empty())
			break;
		imageSize = img.size();
		bool found = false;
		std::vector<cv::Point2f>& centers = imagePoints[i_valid];

		for (int scale = 1; scale <= maxScale; scale++)
		{
			cv::Mat timg;
			if (scale == 1)
				timg = img;
			else
				cv::resize(img, timg, cv::Size(), scale, scale);
			
			found = cv::findCirclesGrid(timg, boardSize, centers, cv::CALIB_CB_ASYMMETRIC_GRID);
			if (found)
			{
				if (scale > 1)
				{
					cv::Mat centersMat(centers);
					centersMat *= 1. / scale;
				}
				break;
			}
		}
		if (displayCorners)
		{
			cout << filename << endl;
			cv::Mat cimg, cimg1;
			cvtColor(img, cimg, cv::COLOR_GRAY2BGR);
			drawChessboardCorners(cimg, boardSize, centers, found);
			double sf = 640. / MAX(img.rows, img.cols);
			resize(cimg, cimg1, cv::Size(), sf, sf);
			imshow("corners", cimg1);
			char c = (char)cv::waitKey(500);
			if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
				exit(-1);
		}
		else
			putchar('.');
		
		if (!found) {
			std::cout << "NOT FOUND!" << std::endl;
		}
		else {
			i_valid++;
		}
	}

	cout << i_valid << " images with targets have been successfully detected.\n";
	if (nimages < 2)
	{
		cout << "Error: too few pairs to run the calibration\n";
		return;
	}


	nimages = i_valid;
	imagePoints.resize(nimages);
	objectPoints.resize(nimages);

	int j, k;
	for (i = 0; i < nimages; i++)
	{

		for (int j = 0; j < boardSize.height; j++)
			for (int k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(cv::Point3f( float((2*k + j % 2)*squareSize), float(j*squareSize), 0) );
	}
	std::vector<cv::Mat> rvecs, tvecs;
	std::vector<float> reprojErrs;
	bool accepted = false;
	while (!accepted) {
		std::cout << "Calibrating" << std::endl;
		cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
		double totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints, rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs, false);
		std::cout << "Total average reprojection error: " << totalAvgErr << std::endl;
		double mean, sd;
		sdmean(reprojErrs, sd, mean);
		std::cout << "Error SD: " << sd << std::endl;

		int worst_index = -1;
		double worst_error = 0;
		for (i = 0; i < reprojErrs.size(); i++) {
			std::cout << "Image " << i << " error: " << reprojErrs[i] << std::endl;
			if (reprojErrs[i] - mean > 2 * sd) {
				std::cout << "Error deviates from others more than 2*sdev!" << std::endl;
				if (reprojErrs[i] > worst_error) {
					worst_error = reprojErrs[i];
					worst_index = i;
				}
			}
		}
		if (worst_index != -1) {
			std::cout << "Removing image currently at index: " << worst_index << std::endl;
			objectPoints.erase(objectPoints.begin() + worst_index);
			imagePoints.erase(imagePoints.begin() + worst_index);
		}
		else {
			accepted = true;
		}
	}


}

std::vector<std::vector<std::string>> getpairs(std::string filelistname, std::vector<int> &c1s, std::vector<int> &c2s) {
	std::vector<std::vector<std::string>> pairlist;
	std::vector<std::string> tempfilelist;
	std::vector<std::string> filelist;

	cv::FileStorage fs(filelistname, cv::FileStorage::READ);
	fs["filelist"] >> filelist;
	fs.release();

	int p1 = -1, p2 = -1;
	for (int i = 0; i < filelist.size(); i++) {

		std::string filename = filelist[i].substr(0, filelist[i].rfind('.')); 
		size_t spritPosition1 = filename.rfind('/');
		size_t spritPosition2 = filename.rfind('\\');
		if (spritPosition1 != std::string::npos)
		{
			filename = filename.substr(spritPosition1 + 1, filename.size() - 1);
		}
		else if (spritPosition2 != std::string::npos)
		{
			filename = filename.substr(spritPosition2 + 1, filename.size() - 1);
		}

		int cameraVertex, timestamp;
		if (filename != "pattern") {
			sscanf(filename.c_str(), "%d-%d", &cameraVertex, &timestamp);
			bool newpair = false;
			if (cameraVertex == p1 || cameraVertex == p2) {}
			else if (p1 == -1) {
				p1 = cameraVertex;
				c1s.push_back(p1);
			}
			else if (p2 == -1) {
				p2 = cameraVertex;
				c2s.push_back(p2);
			}
			else {
				pairlist.push_back(tempfilelist);
				tempfilelist.clear();
				p1 = cameraVertex;
				c1s.push_back(p1);
				p2 = -1;
			}

			tempfilelist.push_back(filelist[i]);
		}
	}
	pairlist.push_back(tempfilelist);

	return pairlist;
}

std::vector<std::vector<std::string>> getcameralists(std::string filelistname, std::vector<int> &cameras) {
	std::vector<std::vector<std::string>> cameralists;
	std::vector<std::string> tempfilelist;
	std::vector<std::string> filelist;

	cv::FileStorage fs(filelistname, cv::FileStorage::READ);
	fs["filelist"] >> filelist;
	fs.release();

	int p1 = -1;
	for (int i = 0; i < filelist.size(); i++) {

		std::string filename = filelist[i].substr(0, filelist[i].rfind('.'));
		size_t spritPosition1 = filename.rfind('/');
		size_t spritPosition2 = filename.rfind('\\');
		if (spritPosition1 != std::string::npos)
		{
			filename = filename.substr(spritPosition1 + 1, filename.size() - 1);
		}
		else if (spritPosition2 != std::string::npos)
		{
			filename = filename.substr(spritPosition2 + 1, filename.size() - 1);
		}

		int cameraVertex, timestamp;
		if (filename != "pattern") {
			sscanf(filename.c_str(), "%d-%d", &cameraVertex, &timestamp);
			if (cameraVertex == p1) {
				//do nothing
			}
			else if (p1 == -1) {
				p1 = cameraVertex;
				cameras.push_back(p1);
			}
			else {
				cameralists.push_back(tempfilelist);
				tempfilelist.clear();
				p1 = cameraVertex;
				cameras.push_back(p1);
			}

			tempfilelist.push_back(filelist[i]);
		}
	}
	cameralists.push_back(tempfilelist);

	return cameralists;
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

int main(int argc, char * argv[])
{

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

	cv::FileStorage fs_calibconfig("calibrationconfig.xml", cv::FileStorage::READ);

	cv::FileNode calibrationsets = fs_calibconfig["calibrationsets"];

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
			if (combinationfn.size() == 4)
				csets.back().combinations.back().n = (int)combinationfn["n"];
			else
				csets.back().combinations.back().n = (int)cset["n"];
		}
		cout << csets.back();
	}


	fs_calibconfig.release();

	

	if (all_sets || (selected_set == 0) ){

		int ncameras = csets[0].cameramapping.size();


		std::string filelistfile = "./calibration_set_0/filelist.xml";

		cv::TermCriteria termcriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, 1e-7);
		cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SiftFeatureDetector::create(0, 4, 0.04, 10.0, 1.4);
		cv::Ptr<cv::DescriptorExtractor> descriptor = cv::xfeatures2d::SiftDescriptorExtractor::create(0, 4, 0.04, 10.0, 1.4);

		cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");

		cv::multicalib::MultiCameraCalibration multiCalib(0, ncameras, "./calibration_set_0/filelist.xml", 420, 297, 1, 0, 15, 0, termcriteria, detector, descriptor, matcher);
		std::cout << "Running multicalib on " << ncameras << " cameras" << std::endl;
		double error = multiCalib.run();		
		multiCalib.writeParameters("calibration_set_0.yaml");
		multiCalib.writePoints("calibration_set_0_points.xml");
		std::cout << "\nMean reprojection error: " << error << std::endl;

	}
	if (all_sets || (selected_set == 1)) {
		std::string filelistfile = "./calibration_set_1/filelist.xml";
		std::vector<int> cameras;

		std::vector<std::vector<std::string>> cameralists = getcameralists(filelistfile, cameras);
		
		cv::FileStorage fs_calibration_set_1("calibration_set_1.yaml", cv::FileStorage::WRITE);

		for (int i = 0; i < cameralists.size(); i++) {

			std::string camera_str = cam_int2str(csets[1].cameramapping, cameras[i]);

			cv::Mat camera_matrix, camera_distortion;

			InternalCalib(cameralists[i], cv::Size(4, 11), 20, camera_matrix, camera_distortion);
			
			fs_calibration_set_1 << ("camera_matrix_" + std::to_string(cameras[i])) << camera_matrix;
			fs_calibration_set_1 << ("camera_distortion_" + std::to_string(cameras[i])) << camera_distortion;

		}

		fs_calibration_set_1.release();

	}
	if (all_sets || (selected_set == 2)) {
	
		cv::FileStorage fs_calibration_set_0, fs_calibration_set_1;
		bool intrinsics_available = (boost::filesystem::exists(boost::filesystem::path("calibration_set_0.yaml")) && boost::filesystem::exists(boost::filesystem::path("calibration_set_1.yaml")));
		if (intrinsics_available) {
			std::cout << "Intrinsics are available!" << std::endl;
			fs_calibration_set_0 = cv::FileStorage("calibration_set_0.yaml", cv::FileStorage::READ);
			fs_calibration_set_1 = cv::FileStorage("calibration_set_1.yaml", cv::FileStorage::READ);
		}
		else {
			std::cout << "Intrinsics are NOT available!" << std::endl;
			return 0;
		}

		std::string filelistfile = "./calibration_set_2/filelist.xml";
		std::vector<int> c1s, c2s;

		
		std::vector<std::vector<std::string>> pairlist = getpairs(filelistfile, c2s, c1s);


		cv::FileStorage fs_calibration_set_2("calibration_set_2.yaml", cv::FileStorage::WRITE);
		
		for (int i = 0; i < pairlist.size(); i++) {

			std::cout << "Camera pair: " << c1s[i] << " - " << c2s[i] << std::endl;
			
			std::string c1_str = cam_int2str(csets[2].cameramapping, c1s[i]);
			std::string c2_str = cam_int2str(csets[2].cameramapping, c2s[i]);
			cv::Mat camera_matrix_1, camera_distortion_1, camera_matrix_2, camera_distortion_2;
			
			if (intrinsics_available) {
				int c1_id = cam_str2int(csets[0].cameramapping, c1_str); // This is not clever...
				int c2_id = cam_str2int(csets[1].cameramapping, c2_str);
				std::cout << "Camera 1 string: " << c1_str << " and camera 2 string: " << c2_str << std::endl;
				std::cout << "Camera 1 id: " << c1_id << " and camera 2 id: " << c2_id << std::endl;

				fs_calibration_set_0["camera_matrix_" + std::to_string(c1_id)] >> camera_matrix_1;
				fs_calibration_set_0["camera_distortion_" + std::to_string(c1_id)] >> camera_distortion_1;
				std::cout << "Camera matrix 1: " << camera_matrix_1 << std::endl;

				fs_calibration_set_1["camera_matrix_" + std::to_string(c2_id)] >> camera_matrix_2;
				fs_calibration_set_1["camera_distortion_" + std::to_string(c2_id)] >> camera_distortion_2;
				std::cout << "Camera matrix 2: " << camera_matrix_2 << std::endl;
			}
			else {
				camera_matrix_1 = camera_distortion_1 = camera_matrix_2 = camera_distortion_2 = cv::Mat();
			}

			cv::Mat R, T;

			StereoCalib(pairlist[i], cv::Size(4, 11), 20, camera_matrix_2, camera_distortion_2, camera_matrix_1, camera_distortion_1, R, T);
			fs_calibration_set_2 << ("camera_R_" + std::to_string(c2s[i])) << R;
			fs_calibration_set_2 << ("camera_T_" + std::to_string(c2s[i])) << T;

		}

		fs_calibration_set_2.release();


	}

	if (all_sets || (selected_set == 3)) {

		cv::FileStorage fs_calibration_set_0, fs_calibration_set_1, fs_calibration_set_2;
		bool all_available = (boost::filesystem::exists(boost::filesystem::path("calibration_set_0.yaml")) && 
							  boost::filesystem::exists(boost::filesystem::path("calibration_set_1.yaml")) &&
							  boost::filesystem::exists(boost::filesystem::path("calibration_set_2.yaml")));
		if (all_available) {
			std::cout << "Intrinsics are available!" << std::endl;
			fs_calibration_set_0 = cv::FileStorage("calibration_set_0.yaml", cv::FileStorage::READ);
			fs_calibration_set_1 = cv::FileStorage("calibration_set_1.yaml", cv::FileStorage::READ);
			fs_calibration_set_2 = cv::FileStorage("calibration_set_2.yaml", cv::FileStorage::READ);
		}
		else {
			std::cout << "Not all calibration files found." << std::endl;
			return 0;
		}

		cv::FileStorage fs_calibration_final("calibration_final.yaml", cv::FileStorage::WRITE);

		for (int i = 0; i < csets[0].cameramapping.size(); i++){
			cv::Mat camera_matrix, camera_distortion, camera_pose;
			fs_calibration_set_0["camera_matrix_" + std::to_string(i)] >> camera_matrix;
			fs_calibration_set_0["camera_distortion_" + std::to_string(i)] >> camera_distortion;
			fs_calibration_set_0["camera_pose_" + std::to_string(i)] >> camera_pose;
			
			std::string cam_str = cam_int2str(csets[0].cameramapping, i);
			int id_final = cam_str2int(csets[2].cameramapping, cam_str) + i % 2; // << fix this when you fix the camera IDs
			fs_calibration_final << "camera_matrix_" + std::to_string(id_final) << camera_matrix;
			fs_calibration_final << "camera_distortion_" + std::to_string(id_final) << camera_distortion;
			fs_calibration_final << "camera_pose_" + std::to_string(id_final) << camera_pose;
		}

		//Flir intrinsics and extrinsics
		for (int i = 0; i < csets[1].cameramapping.size(); i++) {
			cv::Mat camera_matrix, camera_distortion, camera_R, camera_T, camera_pose, relative_pose, reference_pose;
			fs_calibration_set_1["camera_matrix_" + std::to_string(i)] >> camera_matrix;
			fs_calibration_set_1["camera_distortion_" + std::to_string(i)] >> camera_distortion;


			std::string cam_str = cam_int2str(csets[1].cameramapping, i);
			int id_final = cam_str2int(csets[2].cameramapping, cam_str);

			fs_calibration_set_2["camera_R_" + std::to_string(id_final)] >> camera_R;
			fs_calibration_set_2["camera_T_" + std::to_string(id_final)] >> camera_T;
			cv::hconcat(camera_R, camera_T, relative_pose);
			cv::Mat filler = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1.0);
			cv::vconcat(relative_pose, filler, relative_pose);
			relative_pose.convertTo(relative_pose, CV_32F);

			std::string ref_str = cam_int2str(csets[2].cameramapping, id_final - 1);
			int ref_id = cam_str2int(csets[0].cameramapping, ref_str) + 1; // <<< fix this when you fix the camera IDs

			fs_calibration_set_0["camera_pose_" + std::to_string(ref_id)] >> reference_pose;

			camera_pose = relative_pose*reference_pose; // check this 

			fs_calibration_final << "camera_matrix_" + std::to_string(id_final) << camera_matrix;
			fs_calibration_final << "camera_distortion_" + std::to_string(id_final) << camera_distortion;
			fs_calibration_final << "camera_pose_" + std::to_string(id_final) << camera_pose;
		}

		fs_calibration_final.release();


	}

	if (all_sets || (selected_set == 5)) {

		struct edge
		{
			int cameraVertex;   
			int timestamp;    
			cv::Mat transform;  
			cv::Mat imagePoints;
			cv::Mat objectPoints;
			cv::Mat measured_z;
			cv::Mat error; 


			edge(int cv, int timestamp_in, cv::Mat trans, cv::Mat imagePoints_in, cv::Mat objectPoints_in)
			{
				cameraVertex = cv;
				timestamp = timestamp_in;
				transform = trans;
				imagePoints = imagePoints_in;
				objectPoints = objectPoints_in;
				measured_z = cv::Mat::zeros(imagePoints_in.size(), CV_32FC1);
				error = cv::Mat::zeros(imagePoints_in.size(), CV_32FC1);
			}
		};

		cv::FileStorage fs("calibration_set_0_points.xml", cv::FileStorage::READ);

		std::vector<edge> edges;
		int nEdges;
		fs["nEdge"] >> nEdges;
		
		std::vector<int> cameras;

		for (int i = 0; i < nEdges; i++) {
			int cv, ts;
			cv::Mat tf, ip, op;
			std::string prefix = "edge_" + std::to_string(i);
			fs[prefix + "_cam"] >> cv;
			fs[prefix + "_photo"] >> ts;
			fs[prefix + "_transform"] >> tf;
			fs[prefix + "_imagePoints"] >> ip;
			fs[prefix + "_objectPoints"] >> op;
			edges.push_back(edge(cv, ts, tf, ip, op));
			cameras.push_back(cv);
		}

		std::vector<int>::iterator it;
		it = std::unique(cameras.begin(), cameras.end());
		cameras.resize(std::distance(cameras.begin(), it));
		std::cout << "Cameras: " << cameras.size() << std::endl;

		std::vector<int> cams;
		std::vector<float> xs;
		std::vector<float> ys;
		std::vector<float> world_x;
		std::vector<float> world_y;
		std::vector<float> world_z;
		std::vector<float> rot_x;
		std::vector<float> rot_y;
		std::vector<float> rot_z;
		std::vector<float> measured_depths;
		std::vector<float> errors;
		std::vector<float> incident_angles;
		std::vector<std::string> filenames;

		int edgeIndex = 0;
		for (int cam : cameras) {
				std::cout << "Camera: " << cam << std::endl;

				double error_sum = 0.0;
				int error_sum_n = 0;

				while (edges[edgeIndex].cameraVertex == cam && edgeIndex < nEdges) {

					std::string depthfilename = "calibration_set_0/" + std::to_string(cam) + "-" + std::to_string(edges[edgeIndex].timestamp) + "_depth.exr";

					if (  (cam % 2 == 0)  && (boost::filesystem::exists(depthfilename)) ) {
						
						cv::Mat depth_image = cv::imread(depthfilename, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH) * 10000;


						cv::Mat op = edges[edgeIndex].objectPoints;
						cv::Mat ip = edges[edgeIndex].imagePoints;
						cv::Mat tf = edges[edgeIndex].transform;
						
						for (int i = 0; i < op.rows; i++) {

							cv::Vec3f point3 = op.at<cv::Vec3f>(i);
							cv::Vec4f point4 = { point3[0], point3[1], point3[2], 1 };

							Eigen::Matrix4f transformation_e;
							cv::cv2eigen(tf, transformation_e);

							Eigen::Vector4f point_e;
							cv::cv2eigen(point4, point_e);

							
							Eigen::Vector4f transformed_point_e = transformation_e*point_e;

							cv::Vec2f image_coords = ip.at<cv::Vec2f>(i);

							float depth = interpolate_z(image_coords, depth_image);
							
							float error;
							if (depth > 0) {
								error = depth - transformed_point_e(2); 
								
								cams.push_back(cam);
								xs.push_back(image_coords[0]);
								ys.push_back(image_coords[1]);
								world_x.push_back(transformed_point_e(0));
								world_y.push_back(transformed_point_e(1));
								world_z.push_back(transformed_point_e(2));
								measured_depths.push_back(depth);
								errors.push_back(error);
								Eigen::Matrix3f rotation  = transformation_e.topLeftCorner(3, 3);
								Eigen::Vector3f point = transformed_point_e.head<3>();
								incident_angles.push_back(incident_angle(point, rotation));
								rot_x.push_back(std::atan2(rotation(2, 1), rotation(2, 2)));
								rot_y.push_back(std::atan2(-rotation(2, 0), std::sqrt(std::pow(rotation(2, 1), 2) + std::pow(rotation(2, 2), 2))));
								rot_z.push_back(std::atan2(rotation(1, 0), rotation(0, 0)));
								filenames.push_back(std::to_string(cam) + "-" + std::to_string(edges[edgeIndex].timestamp) + "_depth.exr");

								error_sum += error;
								error_sum_n++;
								std::cout << "ERROR: " << error << std::endl;
							}
							else {
								depth = 0;
								error = 0;
							}
							edges[edgeIndex].measured_z.at<float>(i) = depth;
							edges[edgeIndex].error.at<float>(i) = error;
						}					
					}
					edgeIndex++;
				}

		}

		cv::FileStorage fs_error("calibration_set_0_errors.xml", cv::FileStorage::WRITE);

		fs_error << "cams" << cams;
		fs_error << "xs" << xs;
		fs_error << "ys" << ys;
		fs_error << "world_x" << world_x;
		fs_error << "world_y" << world_y;
		fs_error << "world_z" << world_z;
		fs_error << "rot_x" << rot_x;
		fs_error << "rot_y" << rot_y;
		fs_error << "rot_z" << rot_z;
		fs_error << "measured_depths" << measured_depths;
		fs_error << "errors" << errors;
		fs_error << "incident_angles" << incident_angles;
		fs_error << "filenames" << filenames;

		fs_error.release();

	}

	if (all_sets || (selected_set == 4)) {


		cv::FileStorage fs("calibration_final.yaml", cv::FileStorage::READ);
		cv::Mat extrinsics;

		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		viewer->setBackgroundColor(1.0, 1.0, 1.0);

		std::vector<pcl::PointXYZ> points;
		points.push_back(pcl::PointXYZ(-10, -10, -10));
		points.push_back(pcl::PointXYZ(10, 10, 10));
		points.push_back(pcl::PointXYZ(10, -10, -10));
		points.push_back(pcl::PointXYZ(-10, 10, -10));
		points.push_back(pcl::PointXYZ(-10, -10, 10));
		points.push_back(pcl::PointXYZ(10, 10, -10));
		points.push_back(pcl::PointXYZ(-10, 10, 10));
		points.push_back(pcl::PointXYZ(10, -10, 10));

		int ncameras = 12;
		std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cam_clouds(ncameras);

		for (int i = 0; i < ncameras; i++) {

			cam_clouds[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

			for (pcl::PointXYZ point : points) {
				cam_clouds[i]->push_back(point);
			}

			cv::Mat transformation(4, 4, CV_32FC1);
			fs["camera_pose_" + std::to_string(i)] >> transformation;
			Eigen::Matrix4f transformation_rotation;
			Eigen::Matrix4f transformation_translation = Eigen::Matrix4f::Identity();

			cv::cv2eigen(transformation, transformation_rotation);
			for (int j = 0; j < 3; j++) {
				transformation_translation(j, 3) = transformation_rotation(j, 3);
				transformation_rotation(j, 3) = 0;
			}

			std::cout << "Translation of camera " << i << ":" << std::endl;
			std::cout << transformation_translation << std::endl;
			std::cout << "Rotation of camera " << i << ":" << std::endl;
			std::cout << transformation_rotation << std::endl;
			std::cout << "Rotation around x: " << std::atan2(transformation_rotation(2, 1), transformation_rotation(2, 2)) / (2 * PI) * 360 << std::endl;
			std::cout << "Rotation around y: " << std::atan2(-transformation_rotation(2, 0), std::sqrt(std::pow(transformation_rotation(2, 1), 2) + std::pow(transformation_rotation(2, 2), 2))) / (2 * PI) * 360 << std::endl;
			std::cout << "Rotation around z: " << std::atan2(transformation_rotation(1, 0), transformation_rotation(0, 0)) / (2 * PI) * 360 << std::endl;

			Eigen::Matrix4f transformation_e;
			cv::cv2eigen(transformation, transformation_e);

			transformation_e = transformation_e.inverse();
			for (auto it = cam_clouds[i]->points.begin(); it != cam_clouds[i]->points.end(); it++) {
				pcl::Vector4fMap pv = it->getVector4fMap();
				pv = transformation_e*pv;
				it->x = pv(0);
				it->y = pv(1);
				it->z = pv(2);
			}

			std::string cloudname = "Camera " + std::to_string(i);
			viewer->addPointCloud(cam_clouds[i], cloudname);
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, cloudname);
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, cloudname, 0);
			viewer->addCoordinateSystem(50, cloudname, 0);
			std::cout << cam_clouds[i]->points[0].x << std::endl;
			std::cout << cam_clouds[i]->points[1].x << std::endl;

		}

		while (!viewer->wasStopped()) {
			viewer->spinOnce();
		}


		std::cout << "Exiting.\n";
		return 1;
	}

}
