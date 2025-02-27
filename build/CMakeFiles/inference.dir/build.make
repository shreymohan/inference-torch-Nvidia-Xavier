# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tecsar/torch_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tecsar/torch_cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/inference.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/inference.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/inference.dir/flags.make

CMakeFiles/inference.dir/inference.cpp.o: CMakeFiles/inference.dir/flags.make
CMakeFiles/inference.dir/inference.cpp.o: ../inference.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tecsar/torch_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/inference.dir/inference.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inference.dir/inference.cpp.o -c /home/tecsar/torch_cpp/inference.cpp

CMakeFiles/inference.dir/inference.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inference.dir/inference.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tecsar/torch_cpp/inference.cpp > CMakeFiles/inference.dir/inference.cpp.i

CMakeFiles/inference.dir/inference.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inference.dir/inference.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tecsar/torch_cpp/inference.cpp -o CMakeFiles/inference.dir/inference.cpp.s

CMakeFiles/inference.dir/inference.cpp.o.requires:

.PHONY : CMakeFiles/inference.dir/inference.cpp.o.requires

CMakeFiles/inference.dir/inference.cpp.o.provides: CMakeFiles/inference.dir/inference.cpp.o.requires
	$(MAKE) -f CMakeFiles/inference.dir/build.make CMakeFiles/inference.dir/inference.cpp.o.provides.build
.PHONY : CMakeFiles/inference.dir/inference.cpp.o.provides

CMakeFiles/inference.dir/inference.cpp.o.provides.build: CMakeFiles/inference.dir/inference.cpp.o


# Object files for target inference
inference_OBJECTS = \
"CMakeFiles/inference.dir/inference.cpp.o"

# External object files for target inference
inference_EXTERNAL_OBJECTS =

inference: CMakeFiles/inference.dir/inference.cpp.o
inference: CMakeFiles/inference.dir/build.make
inference: /home/tecsar/pytorch/torch/lib/libtorch.so
inference: /home/tecsar/pytorch/torch/lib/libc10.so
inference: /usr/lib/aarch64-linux-gnu/libcuda.so
inference: /usr/local/cuda-10.0/lib64/libnvrtc.so
inference: /usr/local/cuda-10.0/lib64/libnvToolsExt.so
inference: /usr/local/cuda-10.0/lib64/libcudart.so
inference: /home/tecsar/pytorch/torch/lib/libc10_cuda.so
inference: /usr/lib/libopencv_dnn.so.3.3.1
inference: /usr/lib/libopencv_ml.so.3.3.1
inference: /usr/lib/libopencv_objdetect.so.3.3.1
inference: /usr/lib/libopencv_shape.so.3.3.1
inference: /usr/lib/libopencv_stitching.so.3.3.1
inference: /usr/lib/libopencv_superres.so.3.3.1
inference: /usr/lib/libopencv_videostab.so.3.3.1
inference: /home/tecsar/pytorch/torch/lib/libc10_cuda.so
inference: /usr/local/cuda/lib64/libnvToolsExt.so
inference: /usr/local/cuda/lib64/libcudart.so
inference: /home/tecsar/pytorch/torch/lib/libcaffe2.so
inference: /home/tecsar/pytorch/torch/lib/libc10.so
inference: /usr/local/cuda-10.0/lib64/libcufft.so
inference: /usr/local/cuda-10.0/lib64/libcurand.so
inference: /usr/lib/aarch64-linux-gnu/libcudnn.so
inference: /usr/local/cuda-10.0/lib64/libcublas.so
inference: /usr/local/cuda-10.0/lib64/libcudart.so
inference: /usr/lib/libopencv_calib3d.so.3.3.1
inference: /usr/lib/libopencv_features2d.so.3.3.1
inference: /usr/lib/libopencv_flann.so.3.3.1
inference: /usr/lib/libopencv_highgui.so.3.3.1
inference: /usr/lib/libopencv_photo.so.3.3.1
inference: /usr/lib/libopencv_video.so.3.3.1
inference: /usr/lib/libopencv_videoio.so.3.3.1
inference: /usr/lib/libopencv_imgcodecs.so.3.3.1
inference: /usr/lib/libopencv_imgproc.so.3.3.1
inference: /usr/lib/libopencv_core.so.3.3.1
inference: CMakeFiles/inference.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tecsar/torch_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable inference"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/inference.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/inference.dir/build: inference

.PHONY : CMakeFiles/inference.dir/build

CMakeFiles/inference.dir/requires: CMakeFiles/inference.dir/inference.cpp.o.requires

.PHONY : CMakeFiles/inference.dir/requires

CMakeFiles/inference.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/inference.dir/cmake_clean.cmake
.PHONY : CMakeFiles/inference.dir/clean

CMakeFiles/inference.dir/depend:
	cd /home/tecsar/torch_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tecsar/torch_cpp /home/tecsar/torch_cpp /home/tecsar/torch_cpp/build /home/tecsar/torch_cpp/build /home/tecsar/torch_cpp/build/CMakeFiles/inference.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/inference.dir/depend

