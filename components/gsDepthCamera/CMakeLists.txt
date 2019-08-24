cmake_minimum_required(VERSION 3.8)

project(gsDepthCamera CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/modules)

set(gsDepthCamera_VERSION_MAJOR 0)
set(gsDepthCamera_VERSION_MINOR 1)

if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif()

find_package(OpenCV REQUIRED)

find_package(Freenect2  REQUIRED)
option(WITH_CUDA "Process Kinect depth frames with CUDA." OFF)
if(${WITH_CUDA})
	add_definitions( -DWITH_CUDA )
endif()

set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost REQUIRED COMPONENTS program_options serialization filesystem)

include_directories(${OPENCV_INCLUDE_DIRS} ${FREENECT2_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

set(SRC gsDepthCamera.cxx)
set(HEAD gsDepthCamera.hxx gsTimer.hxx gsDepthCamera_serialization.hxx)

add_library(gsDepthCamera ${HEAD} ${SRC})
target_link_libraries(gsDepthCamera ${OpenCV_LIBS} ${FREENECT2_LIBRARY} ${Boost_LIBRARIES})

set(SRC_test_display gsDepthCamera_test_display.cxx)
add_executable(gsDepthCamera_test_display ${SRC_test_display})
target_link_libraries(gsDepthCamera_test_display  ${OpenCV_LIBS} ${FREENECT2_LIBRARY} ${Boost_LIBRARIES} gsDepthCamera)

set(SRC_test_dump gsDepthCamera_test_dump.cxx)
add_executable(gsDepthCamera_test_dump ${SRC_test_dump})
target_link_libraries(gsDepthCamera_test_dump  ${OpenCV_LIBS} ${FREENECT2_LIBRARY} ${Boost_LIBRARIES} gsDepthCamera)

set(SRC_test_load gsDepthCamera_test_load.cxx)
add_executable(gsDepthCamera_test_load ${SRC_test_load})
target_link_libraries(gsDepthCamera_test_load  ${OpenCV_LIBS} ${FREENECT2_LIBRARY} ${Boost_LIBRARIES} gsDepthCamera)


message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
message(STATUS "Freenect2 libraries: ${FREENECT2_LIBRARY}")
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "default install path" FORCE )
endif()

install (TARGETS gsDepthCamera DESTINATION "install/lib")
install (FILES gsDepthCamera.hxx DESTINATION "install/include")
