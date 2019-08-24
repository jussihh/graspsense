cmake_minimum_required(VERSION 3.8)

project(gs CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/modules)

# The version number
set(gsDepthCamera_VERSION_MAJOR 0)
set(gsDepthCamera_VERSION_MINOR 1)

if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif()

find_package(PhoXi REQUIRED CONFIG PATHS "$ENV{PHOXI_CONTROL_PATH}")
find_package(PhoLocalization REQUIRED CONFIG PATHS "$ENV{PHO_LOCALIZATION_PATH}")


find_package(OpenCV REQUIRED)

find_package(Freenect2  REQUIRED)
find_package(ebus REQUIRED)

option(WITH_CUDA "Process Kinect depth frames with CUDA." ON)
if(${WITH_CUDA})
	add_definitions( -DWITH_CUDA )
endif()

find_package(PCL REQUIRED)

find_package(glm REQUIRED)

set(GLM_INCLUDE_DIR "C:/Program Files/glm/include")

find_package(glfw3 REQUIRED)
set(GLFW_INCLUDE_DIR "C:/Program Files/glfw3/include")
set(GLFW_LIB "C:/Program Files/glfw3/lib/glfw3.lib")


find_package(Eigen3 REQUIRED NO_MODULE)

set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost REQUIRED program_options serialization filesystem)


set(gsIRCamera_dir ${CMAKE_CURRENT_SOURCE_DIR}\\components\\gsIRCamera\\build\\install)
set(gsDepthCamera_dir ${CMAKE_CURRENT_SOURCE_DIR}\\components\\gsDepthCamera\\build\\install)
set(gsModel_dir ${CMAKE_CURRENT_SOURCE_DIR}\\components\\gsModel\\build\\install)

set(GS_INCLUDE_DIRS ${gsIRCamera_dir}\\include ${gsDepthCamera_dir}\\include ${gsModel_dir}\\include)
set(GS_LIBS 
optimized ${gsIRCamera_dir}\\lib\\gsIRCamera.lib 
optimized ${gsDepthCamera_dir}\\lib\\gsDepthCamera.lib 
optimized ${gsModel_dir}\\lib\\gsModel.lib 
debug ${gsIRCamera_dir}\\lib\\gsIRCamerad.lib 
debug ${gsDepthCamera_dir}\\lib\\gsDepthCamerad.lib
debug ${gsModel_dir}\\lib\\gsModeld.lib
)


set(PHOXI_LIB optimized "C:/Program Files/PhotoneoPhoXiControl/API/lib/PhoXi_API_msvc14_Release_1.2.2.lib" debug "C:/Program Files/PhotoneoPhoXiControl/API/lib/PhoXi_API_msvc14_Debug_1.2.2.lib")
set(PHOXI_LOCA_LIB optimized "C:/Program Files/PhotoneoSDK/Localization/lib/PhoLocalization_SDK_msvc12_Release_1.2.1.lib" debug "C:/Program Files/PhotoneoSDK/Localization/lib/PhoLocalization_SDK_msvc12_Release_1.2.1.lib")


include_directories(${OPENCV_INCLUDE_DIRS} ${EBUS_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${GS_INCLUDE_DIRS} ${FREENECT2_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${GLFW_INCLUDE_DIR} ${GLM_INCLUDE_DIR} ${PHOXI_INCLUDE_DIRS} ${PHOLOCALIZATION_INCLUDE_DIRS})

set(SRC_calibrate gs_calibrate.cxx)
add_executable(gs_calibrate ${SRC_calibrate})
target_link_libraries(gs_calibrate  ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PCL_LIBRARIES})

set(SRC_calibrate_rec gs_calibrate_rec.cxx)
add_executable(gs_calibrate_rec ${SRC_calibrate_rec})
target_link_libraries(gs_calibrate_rec  ${OpenCV_LIBS} ${EBUS_LIBRARIES} ${FREENECT2_LIBRARY} ${Boost_LIBRARIES} ${PHOXI_LIB} ${GS_LIBS} )

set(SRC_syncr gs_syncr.cxx)
add_executable(gs_syncr ${SRC_syncr})
target_link_libraries(gs_syncr ${OpenCV_LIBS} ${EBUS_LIBRARIES} ${FREENECT2_LIBRARY} ${Boost_LIBRARIES} ${GS_LIBS})

set(SRC_convert gs_convert.cxx)
add_executable(gs_convert ${SRC_convert})
target_link_libraries(gs_convert ${OpenCV_LIBS} ${Boost_LIBRARIES})

set(SRC_experiment gs_experiment.cxx)
add_executable(gs_experiment ${SRC_experiment})
target_link_libraries(gs_experiment  ${OpenCV_LIBS} ${EBUS_LIBRARIES} ${FREENECT2_LIBRARY} ${Boost_LIBRARIES}  ${PHOXI_LIB} ${PHOXI_LOCA_LIB} ${GS_LIBS})

set(SRC_experiment_loc gs_experiment_loc.cxx)
add_executable(gs_experiment_loc ${SRC_experiment_loc})
target_link_libraries(gs_experiment_loc  ${OpenCV_LIBS} ${EBUS_LIBRARIES} ${FREENECT2_LIBRARY} ${Boost_LIBRARIES}  ${PHOXI_LIB}  ${PHOXI_LOCA_LIB} ${GS_LIBS})


add_library(glad "${GLFW_INCLUDE_DIR}/glad/glad.h" glad.c)
target_include_directories(glad PRIVATE ${GLFW_INCLUDE_DIR})

set(SRC_project_on_model  gs_project_on_model.cxx)
add_executable(gs_project_on_model ${SRC_project_on_model})
target_link_libraries(gs_project_on_model ${OpenCV_LIBS}  ${EBUS_LIBRARIES} ${FREENECT2_LIBRARY} ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${GS_LIBS} ${GLFW_LIB} glad)

message(STATUS "Boost libraries: ${Boost_LIBRARIES}")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "default install path" FORCE )
endif()

install (TARGETS gs_syncr gs_experiment gs_experiment_loc gs_convert gs_calibrate gs_calibrate_rec gs_project_on_model DESTINATION "install\\bin")
