cmake_minimum_required(VERSION 3.8)

project(gsModel CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/modules)

if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif()

set(gsModel_VERSION_MAJOR 0)
set(gsModel_VERSION_MINOR 1)

find_package(OpenCV REQUIRED)

find_package(PCL REQUIRED)

find_package(Eigen3 REQUIRED NO_MODULE)
set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost REQUIRED COMPONENTS program_options serialization filesystem)

include_directories(${OPENCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

set(SRC gsModel.cxx)
set(HEAD gsModel.hxx)

add_library(gsModel ${HEAD} ${SRC})
target_link_libraries(gsModel ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Boost_LIBRARIES})

set(SRC_test gsModel_test.cxx)
add_executable(gsModel_test ${SRC_test})
target_link_libraries(gsModel_test  ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Boost_LIBRARIES} Eigen3::Eigen gsModel)

set(SRC_cloudmaker gsModel_cloudmaker.cxx)
add_executable(gsModel_cloudmaker ${SRC_cloudmaker})
target_link_libraries(gsModel_cloudmaker  ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Boost_LIBRARIES} Eigen3::Eigen gsModel)

set(SRC_depth2cloud gsModel_depth2cloud.cxx)
add_executable(gsModel_depth2cloud ${SRC_depth2cloud})
target_link_libraries(gsModel_depth2cloud  ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Boost_LIBRARIES} Eigen3::Eigen gsModel)


set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "default install path" FORCE )

install (TARGETS gsModel DESTINATION "install/lib")
install (FILES gsModel.hxx DESTINATION "install/include")
