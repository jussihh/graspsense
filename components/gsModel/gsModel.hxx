#pragma once

#include <opencv2/core.hpp>
#include <pcl/common/common.h>
#include <pcl/PolygonMesh.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/range_image/range_image_planar.h>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/ccalib.hpp>

#include <vector>

namespace gsModel {

	void setValue(cv::Mat &datamap, int x2d, int y2d, pcl::PointXYZI &point);
	void setValue(cv::Mat &datamap, int x2d, int y2d, pcl::PointXYZRGB &point);

	template <typename T> inline void depth2cloud(cv::Mat depthmap, cv::Mat camera_matrix, T &output) {
		float qNaN(std::numeric_limits<float>::quiet_NaN());

		float near_limit = 0;
		float far_limit = 10000;
		assert(depthmap.cols*depthmap.rows == (int)output.points.size());

		std::vector<float> x_lut(depthmap.cols);
		std::vector<float> y_lut(depthmap.rows);

		double fx = camera_matrix.at<double>(0, 0);
		double cx = camera_matrix.at<double>(0, 2);
		double fy = camera_matrix.at<double>(1, 1);
		double cy = camera_matrix.at<double>(1, 2);

		for (int i = 0; i < depthmap.cols; i++) {
			x_lut[i] = (i - cx + 0.5) / fx;
		}
		for (int i = 0; i < depthmap.rows; i++) {
			y_lut[i] = (i - cy + 0.5) / fy;
		}

		int point = 0;
		float depth;

		for (int i = 0; i < depthmap.rows; i++) {
			for (int j = 0; j < depthmap.cols; j++) {
				depth = depthmap.at<float>(i, j);

				if ((depth >= near_limit) && (depth <= far_limit)) {
					output.points[point].z = depth;
					output.points[point].x = x_lut[j] * output.points[point].z;
					output.points[point].y = y_lut[i] * output.points[point].z;
				}
				else {
					std::cout << "QNAN !!!" << std::endl;
					output.points[point].z = qNaN;
					output.points[point].x = qNaN;
					output.points[point].y = qNaN;
				}
				point++;
			}
		}

	}
	template <typename T> inline void depth2cloud(cv::Mat depthmap, cv::Mat camera_matrix_depth, cv::Mat camera_pose_depth, cv::Mat datamap, cv::Mat camera_matrix_data, cv::Mat camera_pose_data, T& output)
	{

		float qNaN(std::numeric_limits<float>::quiet_NaN());
		int w = depthmap.cols;
		int h = depthmap.rows;

		assert(w*h == (int)output.points.size());

		pcl::PointCloud<pcl::PointXYZ> cloud(w, h);
		depth2cloud(depthmap, camera_matrix_depth, cloud);

		pcl::copyPointCloud(cloud, output);

		Eigen::Matrix4f pose_depth, transformation_depth;

		cv::cv2eigen(camera_pose_depth, pose_depth);
		transformation_depth = pose_depth.inverse();

		pcl::transformPointCloud(output, output, transformation_depth);

		double fx = camera_matrix_data.at<float>(0, 0);
		double cx = camera_matrix_data.at<float>(0, 2);
		double fy = camera_matrix_data.at<float>(1, 1);
		double cy = camera_matrix_data.at<float>(1, 2);



		cv::Mat translation_data, rotation_data;

		rotation_data = camera_pose_data(cv::Range(0, 3), cv::Range(0, 3));
		translation_data = camera_pose_data(cv::Range(0, 3), cv::Range(3, 4));

		std::vector<cv::Point3d> cv3dpoints(output.points.size());
		for (int i = 0; i < output.points.size(); i++) {
			cv3dpoints[i].x = output.points[i].x;
			cv3dpoints[i].y = output.points[i].y;
			cv3dpoints[i].z = output.points[i].z;
		}

		cv::Mat rotation_data_v;
		cv::Rodrigues(rotation_data, rotation_data_v);

		std::vector<cv::Point2d> imagepoints(output.points.size());
		cv::projectPoints(cv3dpoints, rotation_data_v, translation_data, camera_matrix_data, cv::Mat(), imagepoints);

		int index = 0;
		int x2d, y2d;

		cv::Size datasize = datamap.size();

		for (int i = 0; i < output.height; i++) {
			for (int j = 0; j < output.width; j++) {
				y2d = static_cast<int>(std::round(imagepoints[index].y));
				x2d = static_cast<int>(std::round(imagepoints[index].x));
				bool inside = y2d >= 0 && y2d < datasize.height && x2d >= 0 && x2d < datasize.width;
				if (inside) {
					setValue(datamap, x2d, y2d, output.points[index]);
				}
				else {
					output.points[index].z = 0;
				}
				index++;
			}
		}

	}

	template <typename T> inline void cropCloud(pcl::PointCloud<T> &inputcloud, pcl::PointCloud<T> &outputcloud, Eigen::Vector4f minVec, Eigen::Vector4f maxVec) {

		pcl::CropBox<T> cropper(new pcl::CropBox<T>);
		cropper.setMax(maxVec);
		cropper.setMin(minVec);
		cropper.setInputCloud(inputcloud.makeShared());
		cropper.filter(outputcloud);
	}

template <class T>
void transform(T &pointcloud, cv::Mat extrinsic_matrix);

}
