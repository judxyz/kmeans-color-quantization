cmake_minimum_required(VERSION 3.20)
project(color_quantization LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenCV_DIR "C:/opencv/build")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui)

option(USE_OPENMP "Build with OpenMP" ON)

add_executable(color_quantization src/b_openmp_kmeans.cpp)
target_link_libraries(color_quantization PRIVATE ${OpenCV_LIBS})
target_include_directories(color_quantization PRIVATE ${OpenCV_INCLUDE_DIRS})

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(color_quantization PRIVATE OpenMP::OpenMP_CXX)
endif()