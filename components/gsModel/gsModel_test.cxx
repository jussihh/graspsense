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


int main(int argc, char * argv[])
{

	double pi = 3.14159265359;

	cv::FileStorage cconfig_fs("calibrationconfig.xml", cv::FileStorage::READ);
	cv::FileNode calibrationsets = cconfig_fs["calibrationsets"];
	std::vector<std::string> cameramapping;
	for (cv::FileNode cset : calibrationsets) {
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

	std::string experimentname = "data";
	std::vector<boost::filesystem::path> depthfiles, rgbfiles;

	get_all(experimentname + "/" + cameramapping[0], ".exr", depthfiles);
	get_all(experimentname + "/" + cameramapping[2], ".jpg", rgbfiles);
	
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
	if(1) {
		int nclouds = 1;
		std::vector<pcl::PointCloud<pcl::PointXYZRGB>> clouds(1);
		char filename[100];
		reader.read("onemodelmerged.ply", clouds[0]);


		std::vector<std::string> cloudnames = { "mycloud1", "mycloud2", "mycloud3", "mycloud4" };
		Eigen::Vector4f minVec(-100, -150, 500, 0), maxVec(150, 150, 1000, 0);

		for (int i = 0; i < nclouds; i++) {
			viewer.addPointCloud(clouds[i].makeShared(), cloudnames[i]);
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloudnames[i]);
		}



		Eigen::Matrix4f animationmatrix;
		float T = 0.02;
		double cT = std::cos(T);
		double sT = std::sin(T);
		animationmatrix << cT, -sT, 0, 0,
			sT, cT, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;

		int n = 2*pi / T - 1;
		for (int i = 0; i < n;i++) {
			posematrix = posematrix * animationmatrix;
			Eigen::Affine3f cloudpose;
			cloudpose = posematrix;
			for (int i = 0; i < nclouds; i++) {
				viewer.updatePointCloudPose(cloudnames[i], cloudpose);
			}
			viewer.spinOnce();
			std::sprintf(filename, "modelspin_2_%03d.png", i);
			std::string filename_s = filename;
			viewer.saveScreenshot(filename_s);
			std::this_thread::sleep_for(std::chrono::milliseconds(30));
		}

	}

	for (int i = 999; i < 179; i++) {
		char filename[100];
		std::sprintf(filename, "cloud%03d.ply", i);
		std::string filename_s = filename;
		reader.read("slushbox_animation/" + filename_s, cloud);

		std::string cloudname = "mycloud";

		Eigen::Vector4f minVec(-100, -150, 500, 0), maxVec(150, 150, 1000, 0);
		viewer.addPointCloud(cloud.makeShared(), cloudname);



		Eigen::Matrix4f animationmatrix;
		float T = 0.01;
		double cT = std::cos(T);
		double sT = std::sin(T);
		animationmatrix << cT, -sT, 0, 0,
			sT, cT, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;

		posematrix = posematrix * animationmatrix;
		Eigen::Affine3f cloudpose;
		cloudpose = posematrix;
		viewer.updatePointCloudPose(cloudname, cloudpose);
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloudname);
		viewer.spinOnce();
		std::sprintf(filename, "modelspin%03d.png", i);
		filename_s = filename;
		viewer.saveScreenshot(filename_s);
		std::this_thread::sleep_for(std::chrono::milliseconds(30));

		viewer.removeAllPointClouds();
	}

	for (int i = 999; i < depthfiles.size(); i++) {
		cv::Mat depth_raw, depth_undistorted, rgb_raw, rgb_undistorted;
		depth_raw = cv::imread(depthfiles[i].string(), cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH) *10000.0;
		rgb_raw = cv::imread(rgbfiles[i].string());
		cv::undistort(depth_raw, depth_undistorted, camera_matrix[0], camera_distortion[0]);
		cv::undistort(rgb_raw, rgb_undistorted, camera_matrix[2], camera_distortion[2]);
		double scale_factor = 1.0;
		cv::resize(depth_undistorted, depth_undistorted, cv::Size(), scale_factor, scale_factor, cv::INTER_CUBIC);
		camera_matrix[0] = camera_matrix[0] * scale_factor;

		int w = depth_undistorted.cols;
		int h = depth_undistorted.rows;

		pcl::PointCloud<pcl::PointXYZ> cloud(w, h);
		pcl::PointCloud<pcl::PointXYZRGB> cloudrgb(w, h);
		gsModel::depth2cloud(depth_undistorted, camera_matrix[0], camera_pose[0], rgb_undistorted, camera_matrix[2], camera_pose[2], cloudrgb);
		std::cout << "Points: " << cloudrgb.points.size() << std::endl;

		pcl::PointCloud<pcl::PointXYZRGB> cloud_cropped;
		Eigen::Vector4f minVec(-130, -60, 400, 0), maxVec(150, 200, 1000, 0);
		gsModel::cropCloud(cloudrgb, cloud_cropped, minVec, maxVec);

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
		std::sprintf(filename, "cloud%03d.ply", i);
		std::string filename_s = filename;
		writer.write(filename, cloud_scaled);

		std::string cloudname = "mycloud";
		viewer.addPointCloud(cloud_scaled.makeShared(), cloudname);

		Eigen::Matrix3f cameramatrix;
		cameramatrix << 800, 0, 1280,
			0, 800, 720,
			0, 0, 1;
		Eigen::Matrix4f posematrix;
		posematrix << 1., 0., 0., 0.,
			0., 1., 0., 0.,
			0., 0., 1., 0.,
			0., 0., 0., 1.;

		Eigen::Matrix4f animationmatrix;
		float T = 0.01;
		double cT = std::cos(T);
		double sT = std::sin(T);
		animationmatrix << cT, -sT, 0, 0,
			sT, cT, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;

		Eigen::Affine3f cloudpose;
		cloudpose = posematrix;
		viewer.setSize(1280, 720);
		viewer.setPosition(0, 0);
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloudname);
		viewer.spinOnce();
		posematrix = posematrix * animationmatrix;
		cloudpose = posematrix;
		viewer.removeAllPointClouds();
	}

	for (int i = 999; i < cameramapping.size();) {
		depthfiles.clear();
		rgbfiles.clear();
		get_all(experimentname + "/static/" + cameramapping[i], ".exr", depthfiles);
		get_all(experimentname + "/" + cameramapping[i+2], ".jpg", rgbfiles);

		std::vector<boost::filesystem::path>::const_iterator first = depthfiles.end()-11;
		std::vector<boost::filesystem::path>::const_iterator last = depthfiles.end();
		std::vector<boost::filesystem::path> lastdepths(first, last);

		cv::Mat depth_raw, depth_undistorted, rgb_raw, rgb_undistorted;
		getMedianMat(lastdepths, depth_raw);
		rgb_raw = cv::imread(rgbfiles[rgbfiles.size()-10].string());

		cv::imshow("depth", depth_raw/10000);
		cv::imshow("rgb", rgb_raw);

		cv::undistort(depth_raw, depth_undistorted, camera_matrix[i], camera_distortion[i]);
		cv::undistort(rgb_raw, rgb_undistorted, camera_matrix[i+2], camera_distortion[i+2]);
		double scale_factor = 1.0;
		cv::resize(depth_undistorted, depth_undistorted, cv::Size(), scale_factor, scale_factor, cv::INTER_CUBIC);
		camera_matrix[i] = camera_matrix[i] * scale_factor;

		int w = depth_undistorted.cols;
		int h = depth_undistorted.rows;

		pcl::PointCloud<pcl::PointXYZ> cloud(w, h);

		pcl::PointCloud<pcl::PointXYZRGB> cloudrgb(w, h);
		gsModel::depth2cloud(depth_undistorted, camera_matrix[i], camera_pose[i], rgb_undistorted, camera_matrix[i+2], camera_pose[i+2], cloudrgb);
		std::cout << "Points: " << cloudrgb.points.size() << std::endl;

		pcl::PointCloud<pcl::PointXYZRGB> cloud_cropped;
		Eigen::Vector4f minVec(-180, -110, 400, 0), maxVec(200, 250, 1000, 0);
		gsModel::cropCloud(cloudrgb, cloud_cropped, minVec, maxVec);

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
		viewer.addPointCloud(cloud_scaled.makeShared(), cloudname);

		cv::waitKey(0);


		i = i + 3;
	}



}
