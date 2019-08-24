#include <Windows.h>

#include "gsDepthCamera.hxx"
#include "gsIRCamera.hxx"
#include "gsTimer.hxx"
#include "gsModel.hxx"

#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <vector>
#include <chrono>
#include <string>
#include <thread>      
#include <list>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

void build_opengl_projection_for_intrinsics(Eigen::Matrix4d &frustum, int *viewport, double alpha, double beta, double skew, double u0, double v0, int img_width, int img_height, double near_clip, double far_clip);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);


void read_shader_src(const char *fname, std::vector<char> &buffer);
GLuint load_and_compile_shader(const char *fname, GLenum shaderType);

const unsigned int SCR_WIDTH = 640;
const unsigned int SCR_HEIGHT = 512;

struct PointXYZDiameterWeightSource
{
	PCL_ADD_POINT4D;               
	float diameter;
	float weight;
	int source;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW 
} EIGEN_ALIGN16;                   

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZDiameterWeightSource,
(float, x, x)
(float, y, y)
(float, z, z)
(float, diameter, diameter)
(float, weight, weight)
(int, source, source)
)

int iterative_filter(pcl::PointCloud<PointXYZDiameterWeightSource> &pts, float radius = 0.009);

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

void coutmat4(glm::mat4 matrix, std::string name = "Matrix") {
	std::cout << name  << ":" << std::endl;
	std::cout << std::scientific << std::setprecision(3);
	for (int row = 0; row < 4; row++) {
		for (int col = 0; col < 4; col++) {
			std::cout << matrix[col][row] << "\t";
		}
		std::cout << std::endl;
	}
}

glm::mat4 eigen2glm(Eigen::Matrix4d m) {
	glm::mat4 output(m(0,0), m(1,0), m(2, 0), m(3,0),
		m(0, 1), m(1, 1), m(2, 1), m(3, 1), 
		m(0, 2), m(1, 2), m(2, 2), m(3, 2), 
		m(0, 3), m(1, 3), m(2, 3), m(3, 3)); 
	return output;

}
bool fromCV2GLM(const cv::Mat& cvmat, glm::mat4* glmmat) {
	if (cvmat.cols != 4 || cvmat.rows != 4) {
		std::cout << "Conversion error, sizes do not match" << std::endl;
		return false;
	}
	if (cvmat.type() != CV_32FC1) {
		std::cout << "Conversion error, wrong cvmat type" << std::endl;
		return false;
	}
	memcpy(glm::value_ptr(*glmmat), cvmat.data, 16 * sizeof(float));
	*glmmat = glm::transpose(*glmmat);
	return true;
}

bool fromGLM2CV(const glm::mat4& glmmat, cv::Mat* cvmat) {
	if (cvmat->cols != 4 || cvmat->rows != 4) {
		(*cvmat) = cv::Mat(4, 4, CV_32F);
	}
	memcpy(cvmat->data, glm::value_ptr(glmmat), 16 * sizeof(float));
	*cvmat = cvmat->t();
	return true;
}


void detectfingerprints(std::vector<cv::KeyPoint> &keypoints, std::string impath, std::string outputpath, cv::Mat camera_matrix, cv::Mat camera_distortion) {

	cv::Mat im = cv::imread(impath, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	im = im*0.04 - (273.15+20); 
	im = im / 15.0;
	im = im * 255; 
	std::cout << "Trying to read path: " << impath << std::endl;

	cv::Mat im_undistorted;
	cv::undistort(im, im_undistorted, camera_matrix, camera_distortion);


	cv::Mat im8bit;
	im_undistorted.convertTo(im8bit, CV_8UC1);

	cv::SimpleBlobDetector::Params params;
	
	params.minThreshold = 50;
	params.maxThreshold = 240;
	params.thresholdStep = 3; 
	params.minRepeatability = 3;

	params.filterByColor = true;
	params.blobColor = 255;

	params.filterByArea = true;
	params.minArea = 5 * 5;
	params.maxArea = 30 * 30;

	params.filterByCircularity = true;
	params.minCircularity = 0.3;

	params.filterByConvexity = true;
	params.minConvexity = 0.5;

	params.filterByInertia = true;
	params.minInertiaRatio = 0.1;

	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

	detector->detect(im8bit, keypoints);

	im8bit = (im8bit-100)*(255/155);
	cv::Mat ironman128_clone;
	cv::Mat ironman128(128, 1, CV_8UC3, &iron);
	cv::cvtColor(ironman128, ironman128_clone, CV_BGR2RGB); 
	cv::Mat ironman256;
	cv::resize(ironman128_clone, ironman256, cv::Size(), 1, 2);

	cv::applyColorMap(im8bit, im8bit, ironman256);


	cv::Mat im_with_keypoints;

	cv::drawKeypoints(im8bit, keypoints, im_with_keypoints, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);	

	cv::imwrite(outputpath, im_with_keypoints);
	cv::imshow("keypoints", im_with_keypoints);
}

void combinert(cv::InputArray R, cv::InputArray T, cv::OutputArray transformation) {
	cv::Mat RT;
	cv::Mat row4 = (cv::Mat_<double>(1,4) << 0, 0, 0, 1);
	cv::hconcat(R, T, RT);
	cv::vconcat(RT, row4, transformation);
}

namespace fs = ::boost::filesystem;
using vpi = std::vector<fs::path>::iterator;

void pick_a_file(const fs::path& root, std::string &output)
{
	std::cout << "root: " << root << std::endl;
	std::string ext = ".exr";
	std::vector<fs::path> listing;
	if (!fs::exists(root) || !fs::is_directory(root)) return;
	fs::recursive_directory_iterator it(root);
	fs::recursive_directory_iterator endit;

	while (it != endit)
	{
		if (fs::is_regular_file(*it) && it->path().extension() == ext) listing.push_back(it->path());
		++it;
	}

	if (listing.size() > 30) {
		output = listing[listing.size() - 30].string();
	}

}


int main(int argc, char * argv[])
{

	boost::program_options::options_description desc{ "Options" };
	std::vector<std::string> infiles;
	desc.add_options()
		("help,h", "Help screen")
		("output,o", boost::program_options::value<std::string>(), "Output folder")
		("input,i", boost::program_options::value<std::string>(), "Input folder");

	boost::program_options::variables_map vm;
	boost::program_options::store(parse_command_line(argc, argv, desc), vm);
	boost::program_options::notify(vm);

	if (vm.count("help")) {
		std::cout << desc << '\n';
		return(0);
	}

	std::string inputfolder, outputfolder;

	if (vm.count("input")) {
		inputfolder = vm["input"].as<std::string>();
	}

	if (vm.count("output")) {
		outputfolder = vm["output"].as<std::string>();
	}

	std::vector<std::string> cams;
	cams.push_back("192.168.1.104");
	cams.push_back("192.168.1.101");
	cams.push_back("192.168.1.102");
	cams.push_back("192.168.1.103");


	cv::FileStorage settings;
	settings = cv::FileStorage("gs_project_on_model.yaml", cv::FileStorage::READ);
	std::string datapath, ycbpath, outputpath;
	settings["datapath"] >> datapath;
	settings["ycbpath"] >> ycbpath;
	settings["outputpath"] >> outputpath;
	settings.release();


	pcl::PointCloud<PointXYZDiameterWeightSource> all_contact_points;
	std::string fullpath_experiment, fullpath_ply, fullpath_output;
	fullpath_experiment = datapath + inputfolder;
	fullpath_ply = ycbpath + inputfolder + "/google_64k/nontextured.ply";
	fullpath_output = outputpath + inputfolder;


	for (int i_camera = 0; i_camera < 4; i_camera++) {

		cv::FileStorage fs_calibration_set_1;
		fs_calibration_set_1 = cv::FileStorage("calibration_set_1.yaml", cv::FileStorage::READ);
		cv::Mat camera_matrix, camera_distortion;
		fs_calibration_set_1["camera_matrix_" + std::to_string(i_camera)] >> camera_matrix;
		fs_calibration_set_1["camera_distortion_" + std::to_string(i_camera)] >> camera_distortion;
		fs_calibration_set_1.release();

		std::vector<cv::KeyPoint> keypoints;
		std::string sourcedirname = fullpath_experiment + "/" + cams[i_camera];
		std::string chosenfile;
		
		pick_a_file(sourcedirname, chosenfile);

		std::cout << chosenfile;

		if (chosenfile.size() < 3) {
			std::cout << "No recorded data. Exiting." << std::endl;
			return 0;
		}
		else {
			boost::filesystem::create_directory(fullpath_output);
			boost::filesystem::create_directory(outputpath + "/plys");
		}

		detectfingerprints(keypoints, chosenfile, fullpath_output + "/keypoints_cam" + std::to_string(i_camera) + ".tif", camera_matrix, camera_distortion);

		for (cv::KeyPoint kp : keypoints) {
			std::cout<<  "KP:" << std::endl;
			std::cout << "x: " << kp.pt.x << " y: " << kp.pt.y << std::endl;
		}
		cv::Mat camera_R, camera_T, camera_transform, obj_transform;

		cv::FileStorage fs("calibration_set_2.yaml", cv::FileStorage::READ);
		fs["camera_R_" + std::to_string(i_camera)] >> camera_R;
		fs["camera_T_" + std::to_string(i_camera)] >> camera_T;
		fs.release();
		combinert(camera_R, camera_T, camera_transform);
		std::cout << camera_transform;

		if (!fs::exists(fullpath_experiment + "/localization.yaml")) {
			std::cout << "No localization data. Exiting." << std::endl;
		}

		cv::FileStorage fs2(fullpath_experiment + "/localization.yaml", cv::FileStorage::READ);
		fs2["transform_0"] >> obj_transform;
		fs2.release();

		glm::mat4  camera_transform_glm(0.0), obj_transform_glm(0.0);
		camera_transform.convertTo(camera_transform, CV_32F);
		fromCV2GLM(camera_transform, &camera_transform_glm);
		fromCV2GLM(obj_transform, &obj_transform_glm);

		camera_transform_glm[3][0] = 0.001 * camera_transform_glm[3][0];
		camera_transform_glm[3][1] = 0.001 * camera_transform_glm[3][1];
		camera_transform_glm[3][2] = 0.001 * camera_transform_glm[3][2];

		obj_transform_glm[3][0] = 0.001 * obj_transform_glm[3][0];
		obj_transform_glm[3][1] = 0.001 * obj_transform_glm[3][1];
		obj_transform_glm[3][2] = 0.001 * obj_transform_glm[3][2];

		pcl::PolygonMesh input_mesh;
		pcl::PointCloud<pcl::PointXYZ> input_cloud, input_cloud_t;

		pcl::PLYReader plyreader;
		plyreader.read(fullpath_ply, input_mesh);


		pcl::PCLPointCloud2 input_cloud2;
		input_cloud2 = input_mesh.cloud;

		int n_indices = input_mesh.polygons.size();
		std::vector<unsigned int> vindices;
		for (pcl::Vertices verts : input_mesh.polygons) {
			assert(verts.vertices.size() == 3);
			for (uint32_t vert : verts.vertices) {
				vindices.push_back(vert);
			}
		}

		std::cout << "Float size: " << sizeof(float) << " and GL_FLOAT size:" << sizeof(GL_FLOAT) << std::endl;
		std::cout << "Size of polygons: " << n_indices << " Size of indices: " << vindices.size();

		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Localization", NULL, NULL);
		if (window == NULL)
		{
			std::cout << "Failed to create GLFW window" << std::endl;
			glfwTerminate();
			return -1;
		}
		glfwMakeContextCurrent(window);
		glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			std::cout << "Failed to initialize GLAD" << std::endl;
			return -1;
		}


		int success;
		char infoLog[512];

		int n_obj = 2;
		std::vector<GLuint> vertexShader(n_obj), fragmentShader(n_obj), shaderProgram(n_obj);

		for (int i = 0; i < 1; i++) {
			vertexShader[i] = load_and_compile_shader("shader.vert", GL_VERTEX_SHADER);
			fragmentShader[i] = load_and_compile_shader("shader.frag", GL_FRAGMENT_SHADER);


			shaderProgram[i] = glCreateProgram();
			glAttachShader(shaderProgram[i], vertexShader[i]);
			glAttachShader(shaderProgram[i], fragmentShader[i]);
			glLinkProgram(shaderProgram[i]);
			glUseProgram(shaderProgram[i]); 
			glGetProgramiv(shaderProgram[i], GL_LINK_STATUS, &success);
			if (!success) {
				glGetProgramInfoLog(shaderProgram[i], 512, NULL, infoLog);
				std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
			}
			glDeleteShader(vertexShader[i]);
			glDeleteShader(fragmentShader[i]);

		}

		std::vector<glm::mat4> Projection(n_obj), View(n_obj), Model(n_obj);
		Projection[0] = glm::perspective(glm::radians(120.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, -1.0f, 2.0f);

		float cn = 0.2f;
		float cf = 1.2f;

		glm::mat4 projmat(774, 0, -331, 0,
			0, 772, -256, 0,
			0, 0, cn + cf, cn*cf,
			0, 0, -1.0, 0);

		Projection[0] = glm::transpose(projmat);
		Projection[0] = glm::ortho(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0);
		int vp[4];
		Eigen::Matrix4d frustum;
		build_opengl_projection_for_intrinsics(frustum, vp, 778, 777, 0.0, 335.0, 257.0, 640, 512, cn, cf);
		std::cout << "Frustum: " << std::endl << frustum;


		glm::mat4 frustumglm = eigen2glm(frustum);

		Projection[0] = frustumglm;
		Projection[0][2][3] = -1;
		Projection[0][2][2] = -Projection[0][2][2];

		glm::mat4 m(0.0);
		
		float fx = camera_matrix.at<double>(0, 0);
		float cx = camera_matrix.at<double>(0, 2);
		float fy = camera_matrix.at<double>(1, 1);
		float cy = camera_matrix.at<double>(1, 2);
		float width = 640, height = 512, zfar = 1.2, znear = 0.4;

		m[0][0] = 2.0 * fx / width;
		m[0][1] = 0.0;
		m[0][2] = 0.0;
		m[0][3] = 0.0;

		m[1][0] = 0.0;
		m[1][1] = -2.0 * fy / height;
		m[1][2] = 0.0;
		m[1][3] = 0.0;

		m[2][0] = 1.0 - 2.0 * cx / width;
		m[2][1] = 2.0 * cy / height - 1.0;
		m[2][2] = (zfar + znear) / (znear - zfar);
		m[2][3] = -1.0;

		m[3][0] = 0.0;
		m[3][1] = 0.0;
		m[3][2] = 2.0 * zfar * znear / (znear - zfar);
		m[3][3] = 0.0;
		Projection[0] = m;

		View[0] = glm::inverse(camera_transform_glm);
		glm::mat4 flipyz(1, 0, 0, 0,
			0, -1, 0, 0,
			0, 0, -1, 0,
			0, 0, 0, 1);
		glm::mat4 flipz(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, -1, 0,
			0, 0, 0, 1);
		glm::mat4 flipy(1, 0, 0, 0,
			0, -1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		glm::mat4 id(1.0);
		View[0] = flipz*View[0];

		Model[0] = obj_transform_glm;
		glm::mat4 rotY = glm::rotate(id, glm::radians(180.0f), glm::vec3(0, 1, 0));
		std::vector<glm::mat4> mvp(n_obj);
		mvp[0] = Projection[0] * View[0] * Model[0]; 


		coutmat4(Projection[0], "Projection");
		coutmat4(View[0], "View");
		coutmat4(camera_transform_glm, "Cam trans");
		coutmat4(Model[0], "Model");
		coutmat4(View[0] * Model[0], "Modelview");
		coutmat4(mvp[0], "MVP");

		
		GLuint MMatrixID = glGetUniformLocation(shaderProgram[0], "M");
		GLuint VMatrixID = glGetUniformLocation(shaderProgram[0], "V");
		GLuint PMatrixID = glGetUniformLocation(shaderProgram[0], "P");
		GLenum err;

		while ((err = glGetError()) != GL_NO_ERROR) {
			cerr << "OpenGL error stage t-2: " << std::hex << err << endl;
		}

		glUniformMatrix4fv(MMatrixID, 1, GL_FALSE, &Model[0][0][0]);
		glUniformMatrix4fv(VMatrixID, 1, GL_FALSE, &View[0][0][0]);
		glUniformMatrix4fv(PMatrixID, 1, GL_FALSE, &Projection[0][0][0]);

		while ((err = glGetError()) != GL_NO_ERROR) {
			cerr << "OpenGL error stage t-1: " << std::hex << err << endl;
		}

		int n_vertices = input_mesh.cloud.data.size();


		std::vector<GLuint> VBO(n_obj), VAO(n_obj), EBO(n_obj);

		for (int i = 0; i < n_obj; i++) {

		}

		glGenVertexArrays(1, &VAO[0]);
		glGenBuffers(1, &VBO[0]);
		glGenBuffers(1, &EBO[0]);


		glBindVertexArray(VAO[0]);

		glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(input_cloud2.data[0])*input_cloud2.data.size(), &input_cloud2.data[0], GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[0]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(vindices[0])*vindices.size(), &vindices[0], GL_STATIC_DRAW);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (GLvoid*)0); //position (changed *void to GLVoid*
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (GLvoid*)(5*sizeof(float))); //normals
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		std::cout << std::endl << "Sizeof input_mesh.cloud.data: " << sizeof(input_cloud2.data) << " and sizeof(vindices): " << sizeof(vindices) << std::endl;
		while ((err = glGetError()) != GL_NO_ERROR) {
			cerr << "OpenGL error stage t-0: " << std::hex << err << endl;
		}


		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		{
			processInput(window);

			glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);
			glClear(GL_DEPTH_BUFFER_BIT);

			glUseProgram(shaderProgram[0]);
			glBindVertexArray(VAO[0]); 
			glDrawElements(GL_TRIANGLES, vindices.size(), GL_UNSIGNED_INT, 0);
			glfwSwapBuffers(window);
			glfwPollEvents();
			while ((err = glGetError()) != GL_NO_ERROR) {
				cerr << "OpenGL error stage t+1: " << std::hex << err << endl;
			}

			cv::Mat bgrbuffer(SCR_HEIGHT, SCR_WIDTH, CV_8UC3);
			glPixelStorei(GL_PACK_ALIGNMENT, (bgrbuffer.step & 3) ? 1 : 4);
			glPixelStorei(GL_PACK_ROW_LENGTH, bgrbuffer.step / bgrbuffer.elemSize());
			glReadBuffer(GL_FRONT);
			glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_BGR, GL_UNSIGNED_BYTE, bgrbuffer.data);
			cv::flip(bgrbuffer, bgrbuffer, 0);
			cv::imshow("BGR Buffer", bgrbuffer);


			cv::Mat depthbuffer(SCR_HEIGHT, SCR_WIDTH, CV_32FC1);
			glPixelStorei(GL_PACK_ALIGNMENT, (depthbuffer.step & 3) ? 1 : 4);
			glPixelStorei(GL_PACK_ROW_LENGTH, depthbuffer.step / depthbuffer.elemSize());
			glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_DEPTH_COMPONENT, GL_FLOAT, depthbuffer.data);


			cv::flip(depthbuffer, depthbuffer, 0);

			depthbuffer = depthbuffer * 2.0 - 1.0;
			depthbuffer = (2.0 * znear * zfar) / (zfar + znear - depthbuffer * (zfar - znear));

			std::vector<cv::Vec4f> contactpoints;
			std::vector<float> weights;
			int kpcounter = 0;			
			for (cv::KeyPoint kp : keypoints) {
				++kpcounter;
				int x = std::round(kp.pt.x);
				int y = std::round(kp.pt.y);
				float Z = depthbuffer.at<float>(y, x);

				cv::Vec4f contactpoint(0, 0, 0, 0);
				float weight = 0;
				if (Z > (zfar-1e-3)) {
					std::cout << "Keypoint nro: " << kpcounter << " is outside of the object!" << std::endl;
				}
				else {
					float X = Z * (static_cast<float>(x) - cx) / fx;
					float Y = Z * (static_cast<float>(y) - cy) / fy;
					std::cout << "In camera space: X:" << X << " Y:" << Y << " Z:" << Z << std::endl;

					glm::vec4 point(X, Y, Z, 1);
					glm::vec4 point2;
					point2 = glm::inverse(obj_transform_glm) * camera_transform_glm * point;

					contactpoint[0] = point2.x;
					contactpoint[1] = point2.y;
					contactpoint[2] = point2.z;
					contactpoint[3] = Z * kp.size/fx;


					PointXYZDiameterWeightSource pt;

					pt.x = contactpoint[0];
					pt.y = contactpoint[1];
					pt.z = contactpoint[2];
					pt.diameter = contactpoint[3];					

					cv::Mat circle_mask(SCR_HEIGHT, SCR_WIDTH, CV_8UC1, cv::Scalar(0));
					cv::circle(circle_mask, cv::Point(x, y), (int)std::round(0.5*kp.size), 255, -1);
					cv::Scalar meanWeight = cv::mean(bgrbuffer, circle_mask);
					weight = meanWeight[0];
					pt.weight = weight;

					pt.source = i_camera;

					all_contact_points.push_back(pt);

					std::cout << "Keypoint nro: " << kpcounter << std::endl;
					std::cout << "In object space: X:" << point2.x << " Y:" << point2.y << " Z:" << point2.z << " d:" << contactpoint[3] << " Weight: " << weight << std::endl;

				}
				contactpoints.push_back(contactpoint);
				weights.push_back(weight);
			}

			cv::FileStorage outputresults(fullpath_output + "/detections_cam_" + std::to_string(i_camera) +".yaml", cv::FileStorage::WRITE);
			outputresults << "Keypoints" << keypoints;
			outputresults << "Contactpoints" << contactpoints;
			outputresults << "Weights" << weights;
			outputresults.release();

			cv::Mat rgbdepth;
			depthbuffer.convertTo(rgbdepth, CV_8UC3, 255.0);
			cv::imshow("New Depth", rgbdepth);

			cv::imwrite(fullpath_output + "/depth_cam_" + std::to_string(i_camera) + ".exr", depthbuffer);
			cv::imwrite(fullpath_output + "/weight_cam_"   + std::to_string(i_camera) + ".tif", bgrbuffer); //This is now the incident angle map

		} 

		glDeleteVertexArrays(1, &VAO[0]);
		glDeleteBuffers(1, &VBO[0]);
		glDeleteBuffers(1, &EBO[0]);

		glfwTerminate();


	}
	iterative_filter(all_contact_points);

	pcl::PLYWriter writer = pcl::PLYWriter();
	writer.write(fullpath_output + "/filtered_points.ply", all_contact_points, false, false);
	writer.write(outputpath + "/plys/" + inputfolder + ".ply", all_contact_points, false, false);

	std::ofstream myfile;
	myfile.open(outputpath + "/filtered_points.csv", std::ofstream::out | std::ofstream::app);
	for (int i = 0; i < all_contact_points.points.size(); i++) {
		myfile << inputfolder << ",";
		myfile << std::to_string(i) << ",";
		myfile << std::to_string(all_contact_points.points[i].x) << "," << std::to_string(all_contact_points.points[i].x) << "," << std::to_string(all_contact_points.points[i].z) << ",";
		myfile << std::to_string(all_contact_points.points[i].source) << ",";
		myfile << std::to_string(all_contact_points.points[i].diameter) << ",";
		myfile << std::to_string(all_contact_points.points[i].weight) << "\n";
	}
	myfile.close();

	std::cout << "Exiting.\n";
	return 1;

}

int iterative_filter(pcl::PointCloud<PointXYZDiameterWeightSource> &pts, float radius) {
	
	int n_point = pts.points.size();

	std::vector<std::vector<float>> distances(n_point, std::vector<float>(n_point, radius+1.0));
	float min = radius + 1.0;
	int i_min = -1;
	int j_min = -1;
	for (int i = 0; i < n_point; i++) {
		for (int j = i+1; j < n_point; j++) {
			distances[i][j] = pcl::euclideanDistance(pts[i], pts[j]);			
			if (distances[i][j] < min) {
				min = distances[i][j];
				i_min = i;
				j_min = j;
			}			
		}
	}

	std::cout << "Smallest distance is between points: " << i_min << " and " << j_min << std::endl;
	std::cout << "Minimum distance: " << min << std::endl;
	if (min < radius) {
		if (pts[i_min].weight > pts[j_min].weight)
			pts.erase(pts.begin() + j_min);
		else
			pts.erase(pts.begin() + i_min);

		return(iterative_filter(pts, radius) + 1);
	}
	else {
		return 0;
	}
	
}

void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}


// Source: http://jamesgregson.blogspot.com/2011/11/matching-calibrated-cameras-with-opengl.html
void build_opengl_projection_for_intrinsics(Eigen::Matrix4d &frustum, int *viewport, double alpha, double beta, double skew, double u0, double v0, int img_width, int img_height, double near_clip, double far_clip) {
	double L = 0;
	double R = img_width;
	double B = 0; // was 0
	double T = img_height; // was img_height

	// near and far clipping planes, these only matter for the mapping from
	// world-space z-coordinate into the depth coordinate for OpenGL
	double N = near_clip;
	double F = far_clip;

	// set the viewport parameters
	viewport[0] = L;
	viewport[1] = B;
	viewport[2] = R - L;
	viewport[3] = T - B;

	// construct an orthographic matrix which maps from projected
	// coordinates to normalized device coordinates in the range
	// [-1, 1].  OpenGL then maps coordinates in NDC to the current
	// viewport
	Eigen::Matrix4d ortho = Eigen::Matrix4d::Zero();
	ortho(0, 0) = 2.0 / (R - L); 
	ortho(0, 3) = -(R + L) / (R - L); 
	ortho(1, 1) = 2.0 / (T - B); 
	ortho(1, 3) = -(T + B) / (T - B);
	ortho(2, 2) = -2.0 / (F - N); 
	ortho(2, 3) = -(F + N) / (F - N);
	ortho(3, 3) = 1.0;

	// construct a projection matrix, this is identical to the 
	// projection matrix computed for the intrinsicx, except an
	// additional row is inserted to map the z-coordinate to
	// OpenGL. 
	Eigen::Matrix4d tproj = Eigen::Matrix4d::Zero();
	tproj(0, 0) = alpha; tproj(0, 1) = skew; tproj(0, 2) = u0;
	tproj(1, 1) = beta; tproj(1, 2) = v0;
	tproj(2, 2) = -(N + F); tproj(2, 3) = -N*F;
	tproj(3, 2) = 1.0;

	// resulting OpenGL frustum is the product of the orthographic
	// mapping to normalized device coordinates and the augmented
	// camera intrinsic matrix
	frustum = ortho*tproj;
}

// Source: https://github.com/sol-prog/OpenGL-101
// store the shader source in a std::vector<char>
void read_shader_src(const char *fname, std::vector<char> &buffer) {
	std::ifstream in;
	in.open(fname, std::ios::binary);

	if (in.is_open()) {
		// Get the number of bytes stored in this file
		in.seekg(0, std::ios::end);
		size_t length = (size_t)in.tellg();

		// Go to start of the file
		in.seekg(0, std::ios::beg);

		// Read the content of the file in a buffer
		buffer.resize(length + 1);
		in.read(&buffer[0], length);
		in.close();
		// Add a valid C - string end
		buffer[length] = '\0';
	}
	else {
		std::cerr << "Unable to open " << fname << " I'm out!" << std::endl;
		exit(-1);
	}
}
// Source: https://github.com/sol-prog/OpenGL-101
// Compile a shader
GLuint load_and_compile_shader(const char *fname, GLenum shaderType) {
	// Load a shader from an external file
	std::vector<char> buffer;
	read_shader_src(fname, buffer);
	const char *src = &buffer[0];

	// Compile the shader
	GLuint shader = glCreateShader(shaderType);
	glShaderSource(shader, 1, &src, NULL);
	glCompileShader(shader);
	// Check the result of the compilation
	GLint test;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &test);
	if (!test) {
		std::cerr << "Shader compilation failed with this message:" << std::endl;
		std::vector<char> compilation_log(512);
		glGetShaderInfoLog(shader, compilation_log.size(), NULL, &compilation_log[0]);
		std::cerr << &compilation_log[0] << std::endl;
		glfwTerminate();
		exit(-1);
	}
	return shader;
}