cmake_minimum_required(VERSION "3.19")
project(computer_vision)
# [optinal] set OpenCV_DIR environment variable
# set(OpenCV_DIR ~/Programs/opencv-4.5/install)
set(CMAKE_CXX_STANDARD 17)
# configure the necessary common CMake environment variables
# needed to include and link the OpenCV program into this
# demo project, namely OpenCV_INCLUDE_DIRS and OpenCV_LIBS
find_package(OpenCV REQUIRED)
# tell the build to include the headers from OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
# include all cpp src files
file(GLOB SOURCES "src/*.cpp")
# specify the executable target to be built
add_executable(main.app ${SOURCES})
# tell it to link the executable target against OpenCV
target_link_libraries(main.app ${OpenCV_LIBS})
