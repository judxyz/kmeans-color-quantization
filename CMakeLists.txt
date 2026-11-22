cmake_minimum_required(VERSION 3.20)
project(color_quantization LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenCV_DIR "C:/opencv/build")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui)

option(USE_OPENMP "Build with OpenMP" ON)

# add_executable(cv2_kmeans prototypes/cv2.cpp)
# target_link_libraries(cv2_kmeans PRIVATE ${OpenCV_LIBS})
# target_include_directories(cv2_kmeans PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(color_quantization src/b_openmp_kmeans.cpp)
target_link_libraries(color_quantization PRIVATE ${OpenCV_LIBS})
target_include_directories(color_quantization PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(benchmark_kmeans benchmark_kmeans.cpp src/b_openmp_kmeans.cpp)
target_compile_definitions(benchmark_kmeans PRIVATE BENCHMARK)
target_link_libraries(benchmark_kmeans PRIVATE ${OpenCV_LIBS})
target_include_directories(benchmark_kmeans PRIVATE ${OpenCV_INCLUDE_DIRS})


if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(color_quantization PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(benchmark_kmeans PRIVATE OpenMP::OpenMP_CXX)

    
endif()