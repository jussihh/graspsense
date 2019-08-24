/*

*/
#pragma once

#include "gsModel.hxx"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ccalib.hpp>

#include <chrono>
#include <string>
#include <thread>      
#include <list>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>

namespace fs = ::boost::filesystem;
using vpi = std::vector<fs::path>::iterator;

void get_all(const fs::path& root, const std::string& ext, std::vector<fs::path>& ret)
{
	if (!fs::exists(root) || !fs::is_directory(root)) return;
	fs::recursive_directory_iterator it(root);
	fs::recursive_directory_iterator endit;

	while (it != endit)
	{
		if (fs::is_regular_file(*it) && it->path().extension() == ext) ret.push_back(it->path());
		++it;
	}

}


void getMedianMat(std::vector<boost::filesystem::path> depthfilenames, cv::Mat &medianmat) {

	int N = depthfilenames.size();

	std::vector<cv::Mat> depths;
	int width, height;

	for (int i = 0; i < N; i++) {
		cv::Mat depth = cv::imread(depthfilenames[i].string(), cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH) *10000.0;
		depths.push_back(depth.clone());
		width = depth.cols;
		height = depth.rows;
		std::cout << "Loading " << depthfilenames[i].string() << std::endl;
	}

	cv::Mat median_depth_mat(height, width, CV_32FC1, NAN);
	std::vector<float> pixelstack(N);
	std::vector<float*> rowptrs(N);

	for (int y = 0; y < height; y++) {
		for (int k = 0; k < N; k++) {
			rowptrs[k] = depths[k].ptr<float>(y);
		}
		float *median_row_ptr = median_depth_mat.ptr<float>(y);

		for (int x = 0; x < width; x++) {
			for (int k = 0; k < N; k++) {
				pixelstack[k] = rowptrs[k][x];
			}
			std::sort(pixelstack.begin(), pixelstack.end());
			median_row_ptr[x] = pixelstack[std::floor(N / 2)];
		}
	}

	medianmat = median_depth_mat.clone();

}

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

bool createdir(std::string dirname) {
	if (boost::filesystem::exists(dirname)) {
		if (!boost::filesystem::is_directory(dirname)) {
			std::cout << "output is a file" << std::endl;
			return 0;
		}
		else {
			std::cout << "Folder exists, overwriting." << std::endl;
			return 1;
		}
	}

	return boost::filesystem::create_directory(dirname);

}

int main(int argc, char * argv[])
{


	std::string experimentname = "data";
	std::string outputfolder = "output";
	int mode = 0;
	boost::program_options::options_description desc{ "Options" };

	desc.add_options()
		("help,h", "Help screen")
		("output,o", boost::program_options::value<std::string>(), "Output folder")
		("input,i", boost::program_options::value<std::string>(), "Input folder")
		("mode,m", boost::program_options::value<int>()->default_value(0), "Mode [0=exr], 1=rgb");


	boost::program_options::variables_map vm;
	boost::program_options::store(parse_command_line(argc, argv, desc), vm);
	boost::program_options::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << '\n';
		return(0);
	}

	if (vm.count("input")) {
		experimentname = vm["input"].as<std::string>();
	}

	if (vm.count("output")) {
		outputfolder = vm["output"].as<std::string>();
	}

	if (vm.count("mode")) {
		mode = vm["mode"].as<int>();
	}

	if (!createdir(outputfolder)) return 0;

	double pi = 3.14159265359;

	cv::FileStorage cconfig_fs("calibrationconfig.xml", cv::FileStorage::READ);
	cv::FileNode calibrationsets = cconfig_fs["calibrationsets"];
	std::vector<std::string> cameramapping, cameramapping_depthcams;
	for (cv::FileNode cset : calibrationsets) {
		std::string type;
		cset["type"] >> type;
		if (type == "system")
			cset["cameramapping"] >> cameramapping_depthcams;
		else
			cset["cameramapping"] >> cameramapping;
	}

	
	int ncam = cameramapping.size();

	cv::FileStorage calibrations_fs("calibration_final.yaml", cv::FileStorage::READ);
	std::vector<cv::Mat> camera_distortion(ncam), camera_matrix(ncam), camera_pose(ncam);

	for (int i = 0; i < ncam; i++) {
		calibrations_fs["camera_matrix_" + std::to_string(i)] >> camera_matrix[i];
		calibrations_fs["camera_distortion_" + std::to_string(i)] >> camera_distortion[i];
		calibrations_fs["camera_pose_" + std::to_string(i)] >> camera_pose[i];
	}


	int ncam_depth = cameramapping_depthcams.size();
	cv::FileStorage depth_calibrations_fs("depth_calibration.xml", cv::FileStorage::READ);
	std::vector<float> depth_intercept(ncam_depth), depth_multiplier(ncam_depth);

	for (int i = 0; i < ncam_depth; i++) {
		if (i % 2 == 0) {
			depth_calibrations_fs["cam" + std::to_string(i)] >> depth_intercept[i];
			depth_calibrations_fs["depth_measured_cam" + std::to_string(i)] >> depth_multiplier[i];
		}
		else {
			depth_intercept[i]  = 0;
			depth_multiplier[i] = 1;
		}
	}

	std::vector<boost::filesystem::path> depthfiles, thermalfiles;
	
	pcl::visualization::PCLVisualizer viewer("3D Viewer");
	viewer.setShowFPS(false);
	viewer.setSize(1280, 720);
	viewer.setPosition(0, 0);
	Eigen::Matrix3f cameramatrix;
	cameramatrix << 1000, 0, 1280,
		0, 1000, 720,
		0, 0, 1;
	Eigen::Matrix4f cameraposematrix;
	float T =  -pi / 3.3;
	double cT = std::cos(T);
	double sT = std::sin(T);
	cameraposematrix << 1, 0, 0, 0,
		0, cT, -sT, -.5,
		0, sT, cT, .3,
		0., 0., 0., 1.;

	viewer.setCameraParameters(cameramatrix, cameraposematrix);

	pcl::PLYWriter writer;
	pcl::PLYReader reader;

	pcl::PointCloud<pcl::PointXYZRGB> cloud, cloud_crop;

	Eigen::Matrix4f posematrix;
	posematrix << 1., 0., 0., 0.,
		0., 1., 0., 0.,
		0., 0., 1., 0.,
		0., 0., 0., 1.;
	

	for (int i = 0; i < cameramapping.size(); i = i + 3) {
		depthfiles.clear();
		thermalfiles.clear();
		get_all(experimentname + "/static/" + cameramapping[i], ".exr", depthfiles);
		if(mode == 0)
			get_all(experimentname + "/" + cameramapping[i+2], ".exr", thermalfiles);
		else
			get_all(experimentname + "/" + cameramapping[i + 2], ".jpg", thermalfiles);
		
		std::cout << "Got all, n: " << thermalfiles.size() << " and " << depthfiles.size() << std::endl;

		if ((thermalfiles.size() == 0) || (depthfiles.size() == 0))
			continue;

		std::vector<boost::filesystem::path> depthfiles2;

		for (int j = 0; j < depthfiles.size(); j++) {
			std::string fname = depthfiles[j].string();
			if (fname.find("depth") != std::string::npos)
				depthfiles2.push_back(depthfiles[j]);
		}

		
		int mean_depth_n;
		if (mode == 0)
			mean_depth_n = 10;
		else
			mean_depth_n = 1;

		std::vector<boost::filesystem::path>::const_iterator first = depthfiles2.end() - mean_depth_n;
		std::vector<boost::filesystem::path>::const_iterator last  = depthfiles2.end();
		std::vector<boost::filesystem::path> lastdepths(first, last);

		cv::Mat depth_raw, depth_undistorted, thermal_raw, thermal_undistorted;
		getMedianMat(lastdepths, depth_raw);
		int depthcami = cam_str2int(cameramapping_depthcams, cam_int2str(cameramapping, i));
		float intercept = depth_intercept[depthcami];
		float multiplier = depth_multiplier[depthcami];
		depth_raw = depth_raw*multiplier + intercept;

		std::cout << "\nCorrecting depth values with intercept " << intercept << " and multiplier " << multiplier << std::endl;

		if (mode == 0) {
			thermal_raw = cv::imread(thermalfiles[thermalfiles.size() - 10].string(), cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH) * pow(2, 16)*0.04;;
			thermal_raw = (thermal_raw - 273.15 - 25) / 10;
			cv::Mat thermal_rgb;
			thermal_raw.convertTo(thermal_rgb, CV_8UC3, 255);

			cv::imshow("depth", depth_raw / 10000);
			cv::imshow("thermal", thermal_rgb);
		}
		else {
			thermal_raw = cv::imread(thermalfiles[thermalfiles.size() - 1].string());
		}
	


		cv::undistort(depth_raw, depth_undistorted, camera_matrix[i], camera_distortion[i]);
		cv::undistort(thermal_raw, thermal_undistorted, camera_matrix[i+2], camera_distortion[i+2]);
		double scale_factor = 1.0;
		cv::resize(depth_undistorted, depth_undistorted, cv::Size(), scale_factor, scale_factor, cv::INTER_CUBIC);
		camera_matrix[i] = camera_matrix[i] * scale_factor;

		int w = depth_undistorted.cols;
		int h = depth_undistorted.rows;

		pcl::PointCloud<pcl::PointXYZ> cloud(w, h);

		if (mode == 0) {

			pcl::PointCloud<pcl::PointXYZI> cloudthermal(w, h);

			gsModel::depth2cloud(depth_undistorted, camera_matrix[i], camera_pose[i], thermal_undistorted, camera_matrix[i + 2], camera_pose[i + 2], cloudthermal);
			std::cout << "Points: " << cloudthermal.points.size() << std::endl;

			pcl::PointCloud<pcl::PointXYZI> cloud_cropped;
			Eigen::Vector4f minVec(-200, -200, 400, 0), maxVec(200, 200, 800, 0);
			gsModel::cropCloud(cloudthermal, cloud_cropped, minVec, maxVec);

			pcl::PointCloud<pcl::PointXYZI> cloud_filtered;
			pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
			sor.setInputCloud(cloud_cropped.makeShared());
			sor.setMeanK(50); 
			sor.setStddevMulThresh(1);
			sor.filter(cloud_filtered);
			Eigen::Matrix4f transform = Eigen::Matrix4f::Identity()*0.001;
			pcl::PointCloud<pcl::PointXYZI> cloud_scaled;
			pcl::transformPointCloud(cloud_filtered, cloud_scaled, transform);

			char filename[100];
			std::sprintf(filename, "cloud-model-%03d.ply", i);
			std::string filename_s = filename;
			writer.write(filename, cloud_scaled);

			std::string cloudname = "mycloud" + i;
			pcl::PointCloud<pcl::PointXYZ> thermal_rgb_cloud;
			pcl::copyPointCloud(cloud_scaled, thermal_rgb_cloud);
			viewer.addPointCloud(thermal_rgb_cloud.makeShared(), cloudname);
			viewer.spinOnce();
			cv::waitKey(0);

			i = i + 3;
		}
		else {
			pcl::PointCloud<pcl::PointXYZRGB> cloudthermal(w, h);

			gsModel::depth2cloud(depth_undistorted, camera_matrix[i], camera_pose[i], thermal_undistorted, camera_matrix[i + 2], camera_pose[i + 2], cloudthermal);
			std::cout << "Points: " << cloudthermal.points.size() << std::endl;

			pcl::PointCloud<pcl::PointXYZRGB> cloud_cropped;
			Eigen::Vector4f minVec(-800, -800, 400, 0), maxVec(800, 800, 1000, 0);
			gsModel::cropCloud(cloudthermal, cloud_cropped, minVec, maxVec);

			pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered;
			pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
			sor.setInputCloud(cloud_cropped.makeShared());
			sor.setMeanK(50); 
			sor.setStddevMulThresh(1); 
			sor.filter(cloud_filtered);
			Eigen::Matrix4f transform = Eigen::Matrix4f::Identity()*0.001;
			pcl::PointCloud<pcl::PointXYZRGB> cloud_scaled;
			pcl::transformPointCloud(cloud_filtered, cloud_scaled, transform);

			char filename[100];
			std::sprintf(filename, "cloud-model-%03d.ply", i);
			std::string filename_s = filename;
			writer.write(filename, cloud_scaled);

			std::string cloudname = "mycloud" + i;

			viewer.addPointCloud(cloudthermal.makeShared(), cloudname);
			viewer.spinOnce();
			cv::waitKey(0);			

		}
	}

}
