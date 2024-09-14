cmake_minimum_required(VERSION 2.8)

project(image_classifier)

add_definitions(-std=c++11)
add_definitions("-Wall")

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# ncnn_root_dir is the path to NCNN downloaded. Please change it accordingly.
# include_directories(${OpenCV_DIR}/include)
# link_directories(${OpenCV_DIR}/lib/)
include_directories(${OpenCV_DIR}/include
                    /mnt/c/Users/Admin/Desktop/Puwell/work/ncnn/build/install/include/ncnn)
link_directories(${OpenCV_DIR}/lib/
                 /mnt/c/Users/Admin/Desktop/Puwell/work/ncnn/build/install/lib)

set(OpenCV_LIBS 
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
    opencv_videoio)

add_executable(main main.cpp)
target_link_libraries(main ncnn ${OpenCV_LIBS})
