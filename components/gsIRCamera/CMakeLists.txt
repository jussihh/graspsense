cmake_minimum_required(VERSION 3.8)

project(gsIRCamera CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/modules)

if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif()

set(gsIRCamera_VERSION_MAJOR 0)
set(gsIRCamera_VERSION_MINOR 1)

set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost REQUIRED COMPONENTS program_options serialization filesystem)

find_package(ebus REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS} ${EBUS_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

set(SRC gsIRCamera.cxx)
set(HEAD gsIRCamera.hxx)
add_library(gsIRCamera ${HEAD} ${SRC})
target_link_libraries(gsIRCamera ${OpenCV_LIBS} ${EBUS_LIBRARIES} ${Boost_LIBRARIES})

set(SRC_test_record gsIRCamera_record.cxx)
add_executable(gsIRCamera_record ${SRC_test_record})
target_link_libraries(gsIRCamera_record ${OpenCV_LIBS} ${Boost_LIBRARIES} gsIRCamera)

set(SRC_test_load gsIRCamera_load.cxx)
add_executable(gsIRCamera_load ${SRC_test_load})
target_link_libraries(gsIRCamera_load ${OpenCV_LIBS} ${Boost_LIBRARIES} gsIRCamera)


if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "default install path" FORCE )
endif()

install (TARGETS gsIRCamera DESTINATION "install/lib")
install (FILES gsIRCamera.hxx DESTINATION "install/include")
