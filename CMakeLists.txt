# CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

project(RM_prictice)

#if(EXISTS /usr/bin/clang AND EXISTS /usr/bin/clang++)
#   set(CMAKE_C_COMPILER /usr/bin/clang)
#   set(CMAKE_CXX_COMPILER /usr/bin/clang++)
#endif()

add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME off)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set(CMAKE_PREFIX_PATH "/usr/lib/aarch64-linux-gnu")

# 库文件的放置目录
set(SDK_DIR /home/vertilang/Galaxy_Linux-x86_Gige-U3_32bits-64bits_1.5.2303.9221/Galaxy_camera/lib/x86_64/libgxiapi.so)
# C++11 标准、o3优化、多线程
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -pthread -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

set(OpenCV_DIR "/usr/local/include/opencv4")
# cuda
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/vertilang/TensorRT-8.5.3.1/include)
link_directories(/home/vertilang/TensorRT-8.5.3.1/lib)

find_package(OpenCV  REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(Ceres REQUIRED)

# link the head files
include_directories("/usr/local/include/ceres")
include_directories("/usr/include/eigen3")#new
include_directories(${OpenCV_INCLUDE_DIRS} )
include_directories(${TensorRT_INCLUDE_DIRS})
#include_directories("/usr/src/tensorrt/samples/common")
include_directories(${PROJECT_SOURCE_DIR}/autoaim)
include_directories(${PROJECT_SOURCE_DIR}/DaHeng)
include_directories(${PROJECT_SOURCE_DIR}/common)
include_directories(${PROJECT_SOURCE_DIR}/Mindvision)
include_directories(${PROJECT_SOURCE_DIR}/include)

# link the source files
aux_source_directory(./autoaim SRC_FILES)
aux_source_directory(./DaHeng SRC_FILES)
aux_source_directory(./src SRC_FILES)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
	set(MV_SDK /home/vertilang/linuxSDK_V2.1.0.37/lib/x64/libMVSDK.so)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
	set(MV_SDK /home/vertilang/linuxSDK_V2.1.0.37/lib/x86/libMVSDK.so)
else()
	message(FATAL_ERROR "Unsupport host system architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

cuda_add_executable(${PROJECT_NAME}  ${SRC_FILES} main.cpp )

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})#new
target_link_libraries(${PROJECT_NAME} ${SDK_DIR})
target_link_libraries(${PROJECT_NAME} ${MV_SDK})
target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} cudart)
target_link_libraries(${PROJECT_NAME} fmt::fmt nvonnxparser cuda)



