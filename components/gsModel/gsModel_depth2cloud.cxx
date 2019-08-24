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


	std::string depthmap = "depthmap.exr"; //get this from the arguments later
	std::string outputfolder = "output";
	int cameraID = 0;
	boost::program_options::options_description desc{ "Options" };

	desc.add_options()
		("help,h", "Help screen")
		("output,o", boost::program_options::value<std::string>(), "Output folder")
		("input,i", boost::program_options::value<std::string>(), "Input folder")
	    ("camera,c", boost::program_options::value<int>()->default_value(0), "Camera ID");

	boost::program_options::variables_map vm;
	boost::program_options::store(parse_command_line(argc, argv, desc), vm);
	boost::program_options::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << '\n';
		return(0);
	}

	if (vm.count("input")) {
		depthmap = vm["input"].as<std::string>();
	}

	if (vm.count("output")) {
		outputfolder = vm["output"].as<std::string>();
	}

	if (vm.count("camera")) {
		cameraID = vm["camera"].as<int>();
	}

	double pi = 3.14159265359;

	cv::FileStorage fs_calibration_set_1;
	fs_calibration_set_1 = cv::FileStorage("calibration_set_1.yaml", cv::FileStorage::READ);
	cv::Mat camera_matrix, camera_distortion;
	fs_calibration_set_1["camera_matrix_" + std::to_string(cameraID)] >> camera_matrix;
	fs_calibration_set_1["camera_distortion_" + std::to_string(cameraID)] >> camera_distortion;
	fs_calibration_set_1.release();

	std::cout << "Camera matrix" << std::endl <<  camera_matrix << std::endl;
	std::cout << "Camera distortion" << std::endl << camera_distortion << std::endl;


	pcl::PLYWriter writer;
	pcl::PLYReader reader;



	cv::Mat depth_raw;

	depth_raw = cv::imread(depthmap, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH)*10000;

	int w = depth_raw.cols;
	int h = depth_raw.rows;

	pcl::PointCloud<pcl::PointXYZ> cloud(w, h);
	gsModel::depth2cloud(depth_raw, camera_matrix, cloud);

	char filename[100];
	writer.write("cloud.ply", cloud);


}
