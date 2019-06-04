cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(custom_ops)
set(CMAKE_PREFIX_PATH "/home/tecsar/pytorch/torch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_PREFIX_PATH "/home/tecsar/cnpy/build")
#set(CMAKE_PREFIX_PATH "/home/tecsar/cnpy")
find_package(cnpy REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(inference inference.cpp)

target_link_libraries(inference "${TORCH_LIBRARIES}" ${OpenCV_LIBS} cnpy)

set_property(TARGET inference PROPERTY CXX_STANDARD 11)
