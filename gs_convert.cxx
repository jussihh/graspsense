
#include <vector>
#include <chrono>
#include <string>
#include <thread>      
#include <list>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

static char iron[128][3] = {
	{ 0,   0,  0 },
	{ 0,   0,  0 },
	{ 0,   0,  36 },
	{ 0,   0,  51 },
	{ 0,   0,  66 },
	{ 0,   0,  81 },
	{ 2,   0,  90 },
	{ 4,   0,  99 },
	{ 7,   0, 106 },
	{ 11,   0, 115 },
	{ 14,   0, 119 },
	{ 20,   0, 123 },
	{ 27,   0, 128 },
	{ 33,   0, 133 },
	{ 41,   0, 137 },
	{ 48,   0, 140 },
	{ 55,   0, 143 },
	{ 61,   0, 146 },
	{ 66,   0, 149 },
	{ 72,   0, 150 },
	{ 78,   0, 151 },
	{ 84,   0, 152 },
	{ 91,   0, 153 },
	{ 97,   0, 155 },
	{ 104,   0, 155 },
	{ 110,   0, 156 },
	{ 115,   0, 157 },
	{ 122,   0, 157 },
	{ 128,   0, 157 },
	{ 134,   0, 157 },
	{ 139,   0, 157 },
	{ 146,   0, 156 },
	{ 152,   0, 155 },
	{ 157,   0, 155 },
	{ 162,   0, 155 },
	{ 167,   0, 154 },
	{ 171,   0, 153 },
	{ 175,   1, 152 },
	{ 178,   1, 151 },
	{ 182,   2, 149 },
	{ 185,   4, 149 },
	{ 188,   5, 147 },
	{ 191,   6, 146 },
	{ 193,   8, 144 },
	{ 195,  11, 142 },
	{ 198,  13, 139 },
	{ 201,  17, 135 },
	{ 203,  20, 132 },
	{ 206,  23, 127 },
	{ 208,  26, 121 },
	{ 210,  29, 116 },
	{ 212,  33, 111 },
	{ 214,  37, 103 },
	{ 217,  41,  97 },
	{ 219,  46,  89 },
	{ 221,  49,  78 },
	{ 223,  53,  66 },
	{ 224,  56,  54 },
	{ 226,  60,  42 },
	{ 228,  64,  30 },
	{ 229,  68,  25 },
	{ 231,  72,  20 },
	{ 232,  76,  16 },
	{ 234,  78,  12 },
	{ 235,  82,  10 },
	{ 236,  86,   8 },
	{ 237,  90,   7 },
	{ 238,  93,   5 },
	{ 239,  96,   4 },
	{ 240, 100,   3 },
	{ 241, 103,   3 },
	{ 241, 106,   2 },
	{ 242, 109,   1 },
	{ 243, 113,   1 },
	{ 244, 116,   0 },
	{ 244, 120,   0 },
	{ 245, 125,   0 },
	{ 246, 129,   0 },
	{ 247, 133,   0 },
	{ 248, 136,   0 },
	{ 248, 139,   0 },
	{ 249, 142,   0 },
	{ 249, 145,   0 },
	{ 250, 149,   0 },
	{ 251, 154,   0 },
	{ 252, 159,   0 },
	{ 253, 163,   0 },
	{ 253, 168,   0 },
	{ 253, 172,   0 },
	{ 254, 176,   0 },
	{ 254, 179,   0 },
	{ 254, 184,   0 },
	{ 254, 187,   0 },
	{ 254, 191,   0 },
	{ 254, 195,   0 },
	{ 254, 199,   0 },
	{ 254, 202,   1 },
	{ 254, 205,   2 },
	{ 254, 208,   5 },
	{ 254, 212,   9 },
	{ 254, 216,  12 },
	{ 255, 219,  15 },
	{ 255, 221,  23 },
	{ 255, 224,  32 },
	{ 255, 227,  39 },
	{ 255, 229,  50 },
	{ 255, 232,  63 },
	{ 255, 235,  75 },
	{ 255, 238,  88 },
	{ 255, 239, 102 },
	{ 255, 241, 116 },
	{ 255, 242, 134 },
	{ 255, 244, 149 },
	{ 255, 245, 164 },
	{ 255, 247, 179 },
	{ 255, 248, 192 },
	{ 255, 249, 203 },
	{ 255, 251, 216 },
	{ 255, 253, 228 },
	{ 255, 254, 239 },
	{ 255, 255, 249 },
	{ 255, 255, 249 },
	{ 255, 255, 249 },
	{ 255, 255, 249 },
	{ 255, 255, 249 },
	{ 255, 255, 249 },
	{ 255, 255, 249 },
	{ 255, 255, 249 }
};


#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 
#include <boost/filesystem.hpp>
 
namespace fs = ::boost::filesystem;
using vpi = std::vector<fs::path>::iterator;

int get_mode(cv::Mat &im) {

	int llimit = 7325, hlimit = 7825;

	cv::Mat mask(im.rows, im.cols, CV_8UC1, cv::Scalar(0)), tophalf(im.rows / 2, im.cols, CV_8UC1, cv::Scalar(0)), bottomhalf(im.rows / 2, im.cols, CV_8UC1, cv::Scalar(255));
	cv::vconcat(tophalf, bottomhalf, mask);

	int histSize = hlimit - llimit;
	float range[] = { static_cast<float>(llimit), static_cast<float>(hlimit) };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	cv::Mat hist;
	cv::calcHist(&im, 1, 0, mask, hist, 1, &histSize, &histRange, uniform, accumulate);

	double maxVal = 0;
	cv::Point maxLoc;
	cv::minMaxLoc(hist, 0, &maxVal, 0, &maxLoc);

	return maxLoc.y + llimit;

}

void batchConvertAndSave(const vpi itbegin, const vpi itend, const fs::path outdir, const double scale, const double minK, const double maxK, const int delta = 0, const int mode = 0, const int colormap = 0) {
	std::vector<int> jpeg_params;
	jpeg_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	jpeg_params.push_back(100);

	cv::Mat irimage, ironman128_clone;
	cv::Mat ironman128(128, 1, CV_8UC3, &iron);
	cv::cvtColor(ironman128, ironman128_clone, CV_BGR2RGB); 
	cv::Mat ironman256;	
	cv::resize(ironman128_clone, ironman256, cv::Size(), 1, 2);


	cv::Mat falsecolorimage;
	for (vpi it = itbegin; it < itend; it++) {
		irimage = cv::imread(it->string(), cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
		irimage = irimage - delta;
		cv::Mat im_float;
		irimage.convertTo(im_float, CV_32F, scale);
		im_float = (im_float-minK) / (maxK-minK);
		cv::threshold(im_float, im_float, 0.0, NULL, cv::THRESH_TOZERO);
		cv::threshold(im_float, im_float, 1.0, NULL, cv::THRESH_TRUNC);
		double mini, maxi;
		cv::minMaxLoc(im_float, &mini, &maxi);
		im_float.convertTo(falsecolorimage, CV_8UC3, 255);
		switch (mode) {
		case 0:
			cv::applyColorMap(falsecolorimage, falsecolorimage, ironman256);
			break;
		case 1:
			cv::applyColorMap(falsecolorimage, falsecolorimage, colormap);
			break;
		case 2:
			falsecolorimage = falsecolorimage;
			break;
		}
		std::string outfilename = it->filename().replace_extension(".jpg").string();
		cv::imwrite(outdir.string() + "/" + outfilename, falsecolorimage, jpeg_params);
	}
}
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



int main(int argc, char * argv[])
{
	std::vector<fs::path> input_paths;
	std::string extension = ".exr";
	fs::path output("./converted");

	boost::program_options::options_description desc{ "Options" };
	std::vector<std::string> infiles;
	desc.add_options()
		("help,h", "Help screen")
		("input,i", boost::program_options::value<std::string>()->default_value(""), "Input folder")
		("output,o", boost::program_options::value<std::string>(), "Output folder")
		("mode,m", boost::program_options::value<int>()->default_value(0), "Mode (0=Thermal, 1=Depth, 2=phoxitexture)")
		("colormap,c", boost::program_options::value<int>(), "OpenCV colormap number")
		("minimum,a", boost::program_options::value<float>(), "Minimum temperature (C)")
		("maximum,z", boost::program_options::value<float>(), "Maximum temperature (C)")
		("shift,s", boost::program_options::value<float>(), "Shift temperatures by s (positive)");

	boost::program_options::variables_map vm;
	boost::program_options::store(parse_command_line(argc, argv, desc), vm);
	boost::program_options::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << '\n';
		return(0);
	}

	if (vm.count("input")) {
		fs::path input(vm["input"].as<std::string>());
		std::string fmask = ".exr";

		get_all(input, fmask, input_paths);

		if (input_paths.size() < 1) {
			std::cout << "No input files." << std::endl;
			return 0;
		}

	}

	bool need_directory = true;
	if (vm.count("output")) {
		output = fs::path(vm["output"].as<std::string>());
		if (fs::exists(output)) {
			if (fs::is_directory(output)) {
				std::cout << "Output directory already exists." << std::endl;
				need_directory = false;
			}
			else {
				std::cout << "Output is not a directory." << std::endl;
				return 0;
			}
		}
		else {

		}
	}
	if (need_directory) {
		if (!fs::create_directory(output)) {
			std::cout << "Could not create output directory." << std::endl;
			return 0;
		}
	}

	int colormap = 0;
	if (vm.count("colormap")) {
		colormap = vm["colormap"].as<int>();
		if (colormap < 0 || colormap >12)
			colormap = 0;
	}

	int mode = 0;

	if (vm.count("mode")) {
		mode = vm["mode"].as<int>();
	}

	double minVal, maxVal;

	switch (mode) {
		case 0: 
			minVal = 23;
	 		maxVal = 31;
			break;
		case 1:
			minVal = 500;
			maxVal = 1000;
			break;
		case 2:
			minVal = 0;
			maxVal = 4096;
	}

	double shift = 0;

	if (vm.count("shift")) {
		shift = vm["shift"].as<float>();
	}

	if (vm.count("minimum")) {
		minVal = vm["minimum"].as<float>();
	}
	if (vm.count("maximum")) {
		maxVal = vm["maximum"].as<float>();
	}

	std::vector<std::thread> batchThreads;
	int NTHREADS = 8;

	switch (mode) {
		case 0:
			minVal = 273.15 + minVal + shift;
			maxVal = 273.15 + maxVal + shift;
			break;
		case 1:
		case 2:
			minVal = minVal + shift;
			maxVal = maxVal + shift;		
	}

	assert(minVal < maxVal);


	int chunksize = input_paths.size()/NTHREADS;
	if (chunksize  == 0)
		std::cout << "Problems\n" << std::endl;


	double scale;
	switch (mode) {
	case 0:
		scale = 0.04;
		break;
	case 1:
		scale = 10000;
	case 2:
		scale = 1.0;
	}
	int delta = 0;
	if (mode == 0) {
		
		vpi lastimage = input_paths.end() - 1;
		cv::Mat irimage = cv::imread(lastimage->string(), cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
		int image_mode = get_mode(irimage);
		
		delta = image_mode - 7454;
	}
	cv::FileStorage fs(output.string() + "/delta.xml", cv::FileStorage::WRITE);
	fs << "delta" << delta;
	fs << "image_mode" << (delta + 7454);
	fs.release();

	for (int i = 0; i < NTHREADS; i++) {
		vpi beginit = input_paths.begin() + chunksize*i;
		vpi endit = input_paths.begin() + chunksize*(i+1);
		if (i == (NTHREADS - 1))
			endit = input_paths.end();
		batchThreads.push_back(std::thread(batchConvertAndSave, beginit, endit, output, scale, minVal, maxVal, delta, mode, colormap));
	}

	for (std::thread &thread : batchThreads) {
		if (thread.joinable())
			thread.join();
	}
	std::cout << "Converted " << input_paths.size() << " images." << std::endl;
	return 1;

}
