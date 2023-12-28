cmake_minimum_required(VERSION 2.8)
project(BundleAdjustment)
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -g -march=native")
set(CMAKE_BUILD_TYPE "Release")

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
        "/home/slam/slambook2/slam_code/ba_project/include"
        ${CERES_INCLUDE_DIRS}
        "/usr/include/eigen3/"
        ${OpenCV_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
)



add_executable(ba ./src/ba.cpp)
target_link_libraries(ba ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(g2o_3d2d ./src/g2o_3d2d.cpp)
target_link_libraries(g2o_3d2d ${OpenCV_LIBS} ${CERES_LIBRARIES} g2o_core g2o_stuff)