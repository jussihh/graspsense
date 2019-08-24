
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/range_image/range_image_planar.h>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/ccalib.hpp>

#include <vector>
#include "gsModel.hxx"

namespace gsModel {

	void setValue(cv::Mat &datamap, int x2d, int y2d, pcl::PointXYZI &point) {
		float intensityval = datamap.at<float>(y2d, x2d);
		point.intensity = intensityval;
	}

	void setValue(cv::Mat &datamap, int x2d, int y2d, pcl::PointXYZRGB &point) {
		cv::Vec3b rgbval = datamap.at<cv::Vec3b>(y2d, x2d);
		point.r = static_cast<uint8_t>(rgbval[2]);
		point.g = static_cast<uint8_t>(rgbval[1]);
		point.b = static_cast<uint8_t>(rgbval[0]);
	}
	template <class T>
	void transform(T &pointcloud, cv::Mat extrinsic_matrix) {

	}



}
