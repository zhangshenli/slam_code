cmake_minimum_required(VERSION 2.8)
project(BundleAdjustment)
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -g -march=native")
set(CMAKE_BUILD_TYPE "Release")

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
        "/home/slam/slambook2/ba_project/include"
        ${CERES_INCLUDE_DIRS}
        "/usr/include/eigen3/"
        ${OpenCV_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
)



add_executable(ba ./src/ba.cpp)
target_link_libraries(ba ${OpenCV_LIBS} ${CERES_LIBRARIES})