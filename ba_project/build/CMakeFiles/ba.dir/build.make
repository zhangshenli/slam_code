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
CMAKE_SOURCE_DIR = /home/slam/slambook2/slam_code/ba_project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/slam/slambook2/slam_code/ba_project/build

# Include any dependencies generated for this target.
include CMakeFiles/ba.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ba.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ba.dir/flags.make

CMakeFiles/ba.dir/src/ba.cpp.o: CMakeFiles/ba.dir/flags.make
CMakeFiles/ba.dir/src/ba.cpp.o: ../src/ba.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/slam/slambook2/slam_code/ba_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ba.dir/src/ba.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ba.dir/src/ba.cpp.o -c /home/slam/slambook2/slam_code/ba_project/src/ba.cpp

CMakeFiles/ba.dir/src/ba.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ba.dir/src/ba.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/slam/slambook2/slam_code/ba_project/src/ba.cpp > CMakeFiles/ba.dir/src/ba.cpp.i

CMakeFiles/ba.dir/src/ba.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ba.dir/src/ba.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/slam/slambook2/slam_code/ba_project/src/ba.cpp -o CMakeFiles/ba.dir/src/ba.cpp.s

CMakeFiles/ba.dir/src/ba.cpp.o.requires:

.PHONY : CMakeFiles/ba.dir/src/ba.cpp.o.requires

CMakeFiles/ba.dir/src/ba.cpp.o.provides: CMakeFiles/ba.dir/src/ba.cpp.o.requires
	$(MAKE) -f CMakeFiles/ba.dir/build.make CMakeFiles/ba.dir/src/ba.cpp.o.provides.build
.PHONY : CMakeFiles/ba.dir/src/ba.cpp.o.provides

CMakeFiles/ba.dir/src/ba.cpp.o.provides.build: CMakeFiles/ba.dir/src/ba.cpp.o


# Object files for target ba
ba_OBJECTS = \
"CMakeFiles/ba.dir/src/ba.cpp.o"

# External object files for target ba
ba_EXTERNAL_OBJECTS =

ba: CMakeFiles/ba.dir/src/ba.cpp.o
ba: CMakeFiles/ba.dir/build.make
ba: /usr/local/lib/libopencv_dnn.so.3.4.15
ba: /usr/local/lib/libopencv_highgui.so.3.4.15
ba: /usr/local/lib/libopencv_ml.so.3.4.15
ba: /usr/local/lib/libopencv_objdetect.so.3.4.15
ba: /usr/local/lib/libopencv_shape.so.3.4.15
ba: /usr/local/lib/libopencv_stitching.so.3.4.15
ba: /usr/local/lib/libopencv_superres.so.3.4.15
ba: /usr/local/lib/libopencv_videostab.so.3.4.15
ba: /usr/local/lib/libopencv_viz.so.3.4.15
ba: /usr/local/lib/libceres.a
ba: /usr/local/lib/libopencv_calib3d.so.3.4.15
ba: /usr/local/lib/libopencv_features2d.so.3.4.15
ba: /usr/local/lib/libopencv_flann.so.3.4.15
ba: /usr/local/lib/libopencv_photo.so.3.4.15
ba: /usr/local/lib/libopencv_video.so.3.4.15
ba: /usr/local/lib/libopencv_videoio.so.3.4.15
ba: /usr/local/lib/libopencv_imgcodecs.so.3.4.15
ba: /usr/local/lib/libopencv_imgproc.so.3.4.15
ba: /usr/local/lib/libopencv_core.so.3.4.15
ba: /usr/lib/x86_64-linux-gnu/libglog.so
ba: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1
ba: /usr/lib/x86_64-linux-gnu/libspqr.so
ba: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
ba: /usr/lib/x86_64-linux-gnu/libtbb.so
ba: /usr/lib/x86_64-linux-gnu/libcholmod.so
ba: /usr/lib/x86_64-linux-gnu/libccolamd.so
ba: /usr/lib/x86_64-linux-gnu/libcamd.so
ba: /usr/lib/x86_64-linux-gnu/libcolamd.so
ba: /usr/lib/x86_64-linux-gnu/libamd.so
ba: /usr/lib/x86_64-linux-gnu/liblapack.so
ba: /usr/lib/x86_64-linux-gnu/libblas.so
ba: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
ba: /usr/lib/x86_64-linux-gnu/librt.so
ba: /usr/lib/x86_64-linux-gnu/libcxsparse.so
ba: /usr/lib/x86_64-linux-gnu/liblapack.so
ba: /usr/lib/x86_64-linux-gnu/libblas.so
ba: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
ba: /usr/lib/x86_64-linux-gnu/librt.so
ba: /usr/lib/x86_64-linux-gnu/libcxsparse.so
ba: CMakeFiles/ba.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/slam/slambook2/slam_code/ba_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ba"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ba.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ba.dir/build: ba

.PHONY : CMakeFiles/ba.dir/build

CMakeFiles/ba.dir/requires: CMakeFiles/ba.dir/src/ba.cpp.o.requires

.PHONY : CMakeFiles/ba.dir/requires

CMakeFiles/ba.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ba.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ba.dir/clean

CMakeFiles/ba.dir/depend:
	cd /home/slam/slambook2/slam_code/ba_project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/slam/slambook2/slam_code/ba_project /home/slam/slambook2/slam_code/ba_project /home/slam/slambook2/slam_code/ba_project/build /home/slam/slambook2/slam_code/ba_project/build /home/slam/slambook2/slam_code/ba_project/build/CMakeFiles/ba.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ba.dir/depend

