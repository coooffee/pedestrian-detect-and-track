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
CMAKE_SOURCE_DIR = /home/nvidia/tensorrtx/yolov5/tracking/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/tensorrtx/yolov5/tracking/build

# Include any dependencies generated for this target.
include CMakeFiles/deepsort.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/deepsort.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/deepsort.dir/flags.make

CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.o: CMakeFiles/deepsort.dir/flags.make
CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.o: /home/nvidia/tensorrtx/yolov5/tracking/src/errmsg/errmsg.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/tensorrtx/yolov5/tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.o -c /home/nvidia/tensorrtx/yolov5/tracking/src/errmsg/errmsg.cpp

CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/tensorrtx/yolov5/tracking/src/errmsg/errmsg.cpp > CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.i

CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/tensorrtx/yolov5/tracking/src/errmsg/errmsg.cpp -o CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.s

CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.o.requires:

.PHONY : CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.o.requires

CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.o.provides: CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.o.requires
	$(MAKE) -f CMakeFiles/deepsort.dir/build.make CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.o.provides.build
.PHONY : CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.o.provides

CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.o.provides.build: CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.o


CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.o: CMakeFiles/deepsort.dir/flags.make
CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.o: /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/hungarianoper.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/tensorrtx/yolov5/tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.o -c /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/hungarianoper.cpp

CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/hungarianoper.cpp > CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.i

CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/hungarianoper.cpp -o CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.s

CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.o.requires:

.PHONY : CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.o.requires

CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.o.provides: CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.o.requires
	$(MAKE) -f CMakeFiles/deepsort.dir/build.make CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.o.provides.build
.PHONY : CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.o.provides

CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.o.provides.build: CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.o


CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.o: CMakeFiles/deepsort.dir/flags.make
CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.o: /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/munkres/munkres.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/tensorrtx/yolov5/tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.o -c /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/munkres/munkres.cpp

CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/munkres/munkres.cpp > CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.i

CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/munkres/munkres.cpp -o CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.s

CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.o.requires:

.PHONY : CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.o.requires

CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.o.provides: CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.o.requires
	$(MAKE) -f CMakeFiles/deepsort.dir/build.make CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.o.provides.build
.PHONY : CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.o.provides

CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.o.provides.build: CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.o


CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.o: CMakeFiles/deepsort.dir/flags.make
CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.o: /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/munkres/adapters/adapter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/tensorrtx/yolov5/tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.o -c /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/munkres/adapters/adapter.cpp

CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/munkres/adapters/adapter.cpp > CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.i

CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/munkres/adapters/adapter.cpp -o CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.s

CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.o.requires:

.PHONY : CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.o.requires

CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.o.provides: CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.o.requires
	$(MAKE) -f CMakeFiles/deepsort.dir/build.make CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.o.provides.build
.PHONY : CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.o.provides

CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.o.provides.build: CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.o


CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.o: CMakeFiles/deepsort.dir/flags.make
CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.o: /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/munkres/adapters/boostmatrixadapter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/tensorrtx/yolov5/tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.o -c /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/munkres/adapters/boostmatrixadapter.cpp

CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/munkres/adapters/boostmatrixadapter.cpp > CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.i

CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/tensorrtx/yolov5/tracking/src/thirdPart/munkres/adapters/boostmatrixadapter.cpp -o CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.s

CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.o.requires:

.PHONY : CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.o.requires

CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.o.provides: CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.o.requires
	$(MAKE) -f CMakeFiles/deepsort.dir/build.make CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.o.provides.build
.PHONY : CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.o.provides

CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.o.provides.build: CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.o


CMakeFiles/deepsort.dir/feature/model.cpp.o: CMakeFiles/deepsort.dir/flags.make
CMakeFiles/deepsort.dir/feature/model.cpp.o: /home/nvidia/tensorrtx/yolov5/tracking/src/feature/model.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/tensorrtx/yolov5/tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/deepsort.dir/feature/model.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deepsort.dir/feature/model.cpp.o -c /home/nvidia/tensorrtx/yolov5/tracking/src/feature/model.cpp

CMakeFiles/deepsort.dir/feature/model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deepsort.dir/feature/model.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/tensorrtx/yolov5/tracking/src/feature/model.cpp > CMakeFiles/deepsort.dir/feature/model.cpp.i

CMakeFiles/deepsort.dir/feature/model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deepsort.dir/feature/model.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/tensorrtx/yolov5/tracking/src/feature/model.cpp -o CMakeFiles/deepsort.dir/feature/model.cpp.s

CMakeFiles/deepsort.dir/feature/model.cpp.o.requires:

.PHONY : CMakeFiles/deepsort.dir/feature/model.cpp.o.requires

CMakeFiles/deepsort.dir/feature/model.cpp.o.provides: CMakeFiles/deepsort.dir/feature/model.cpp.o.requires
	$(MAKE) -f CMakeFiles/deepsort.dir/build.make CMakeFiles/deepsort.dir/feature/model.cpp.o.provides.build
.PHONY : CMakeFiles/deepsort.dir/feature/model.cpp.o.provides

CMakeFiles/deepsort.dir/feature/model.cpp.o.provides.build: CMakeFiles/deepsort.dir/feature/model.cpp.o


CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.o: CMakeFiles/deepsort.dir/flags.make
CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.o: /home/nvidia/tensorrtx/yolov5/tracking/src/matching/kalmanfilter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/tensorrtx/yolov5/tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.o -c /home/nvidia/tensorrtx/yolov5/tracking/src/matching/kalmanfilter.cpp

CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/tensorrtx/yolov5/tracking/src/matching/kalmanfilter.cpp > CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.i

CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/tensorrtx/yolov5/tracking/src/matching/kalmanfilter.cpp -o CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.s

CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.o.requires:

.PHONY : CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.o.requires

CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.o.provides: CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.o.requires
	$(MAKE) -f CMakeFiles/deepsort.dir/build.make CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.o.provides.build
.PHONY : CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.o.provides

CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.o.provides.build: CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.o


CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.o: CMakeFiles/deepsort.dir/flags.make
CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.o: /home/nvidia/tensorrtx/yolov5/tracking/src/matching/linear_assignment.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/tensorrtx/yolov5/tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.o -c /home/nvidia/tensorrtx/yolov5/tracking/src/matching/linear_assignment.cpp

CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/tensorrtx/yolov5/tracking/src/matching/linear_assignment.cpp > CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.i

CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/tensorrtx/yolov5/tracking/src/matching/linear_assignment.cpp -o CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.s

CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.o.requires:

.PHONY : CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.o.requires

CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.o.provides: CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.o.requires
	$(MAKE) -f CMakeFiles/deepsort.dir/build.make CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.o.provides.build
.PHONY : CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.o.provides

CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.o.provides.build: CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.o


CMakeFiles/deepsort.dir/matching/nn_matching.cpp.o: CMakeFiles/deepsort.dir/flags.make
CMakeFiles/deepsort.dir/matching/nn_matching.cpp.o: /home/nvidia/tensorrtx/yolov5/tracking/src/matching/nn_matching.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/tensorrtx/yolov5/tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/deepsort.dir/matching/nn_matching.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deepsort.dir/matching/nn_matching.cpp.o -c /home/nvidia/tensorrtx/yolov5/tracking/src/matching/nn_matching.cpp

CMakeFiles/deepsort.dir/matching/nn_matching.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deepsort.dir/matching/nn_matching.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/tensorrtx/yolov5/tracking/src/matching/nn_matching.cpp > CMakeFiles/deepsort.dir/matching/nn_matching.cpp.i

CMakeFiles/deepsort.dir/matching/nn_matching.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deepsort.dir/matching/nn_matching.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/tensorrtx/yolov5/tracking/src/matching/nn_matching.cpp -o CMakeFiles/deepsort.dir/matching/nn_matching.cpp.s

CMakeFiles/deepsort.dir/matching/nn_matching.cpp.o.requires:

.PHONY : CMakeFiles/deepsort.dir/matching/nn_matching.cpp.o.requires

CMakeFiles/deepsort.dir/matching/nn_matching.cpp.o.provides: CMakeFiles/deepsort.dir/matching/nn_matching.cpp.o.requires
	$(MAKE) -f CMakeFiles/deepsort.dir/build.make CMakeFiles/deepsort.dir/matching/nn_matching.cpp.o.provides.build
.PHONY : CMakeFiles/deepsort.dir/matching/nn_matching.cpp.o.provides

CMakeFiles/deepsort.dir/matching/nn_matching.cpp.o.provides.build: CMakeFiles/deepsort.dir/matching/nn_matching.cpp.o


CMakeFiles/deepsort.dir/matching/track.cpp.o: CMakeFiles/deepsort.dir/flags.make
CMakeFiles/deepsort.dir/matching/track.cpp.o: /home/nvidia/tensorrtx/yolov5/tracking/src/matching/track.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/tensorrtx/yolov5/tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/deepsort.dir/matching/track.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deepsort.dir/matching/track.cpp.o -c /home/nvidia/tensorrtx/yolov5/tracking/src/matching/track.cpp

CMakeFiles/deepsort.dir/matching/track.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deepsort.dir/matching/track.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/tensorrtx/yolov5/tracking/src/matching/track.cpp > CMakeFiles/deepsort.dir/matching/track.cpp.i

CMakeFiles/deepsort.dir/matching/track.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deepsort.dir/matching/track.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/tensorrtx/yolov5/tracking/src/matching/track.cpp -o CMakeFiles/deepsort.dir/matching/track.cpp.s

CMakeFiles/deepsort.dir/matching/track.cpp.o.requires:

.PHONY : CMakeFiles/deepsort.dir/matching/track.cpp.o.requires

CMakeFiles/deepsort.dir/matching/track.cpp.o.provides: CMakeFiles/deepsort.dir/matching/track.cpp.o.requires
	$(MAKE) -f CMakeFiles/deepsort.dir/build.make CMakeFiles/deepsort.dir/matching/track.cpp.o.provides.build
.PHONY : CMakeFiles/deepsort.dir/matching/track.cpp.o.provides

CMakeFiles/deepsort.dir/matching/track.cpp.o.provides.build: CMakeFiles/deepsort.dir/matching/track.cpp.o


CMakeFiles/deepsort.dir/matching/tracker.cpp.o: CMakeFiles/deepsort.dir/flags.make
CMakeFiles/deepsort.dir/matching/tracker.cpp.o: /home/nvidia/tensorrtx/yolov5/tracking/src/matching/tracker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/tensorrtx/yolov5/tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/deepsort.dir/matching/tracker.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deepsort.dir/matching/tracker.cpp.o -c /home/nvidia/tensorrtx/yolov5/tracking/src/matching/tracker.cpp

CMakeFiles/deepsort.dir/matching/tracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deepsort.dir/matching/tracker.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/tensorrtx/yolov5/tracking/src/matching/tracker.cpp > CMakeFiles/deepsort.dir/matching/tracker.cpp.i

CMakeFiles/deepsort.dir/matching/tracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deepsort.dir/matching/tracker.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/tensorrtx/yolov5/tracking/src/matching/tracker.cpp -o CMakeFiles/deepsort.dir/matching/tracker.cpp.s

CMakeFiles/deepsort.dir/matching/tracker.cpp.o.requires:

.PHONY : CMakeFiles/deepsort.dir/matching/tracker.cpp.o.requires

CMakeFiles/deepsort.dir/matching/tracker.cpp.o.provides: CMakeFiles/deepsort.dir/matching/tracker.cpp.o.requires
	$(MAKE) -f CMakeFiles/deepsort.dir/build.make CMakeFiles/deepsort.dir/matching/tracker.cpp.o.provides.build
.PHONY : CMakeFiles/deepsort.dir/matching/tracker.cpp.o.provides

CMakeFiles/deepsort.dir/matching/tracker.cpp.o.provides.build: CMakeFiles/deepsort.dir/matching/tracker.cpp.o


CMakeFiles/deepsort.dir/api/deepsort.cpp.o: CMakeFiles/deepsort.dir/flags.make
CMakeFiles/deepsort.dir/api/deepsort.cpp.o: /home/nvidia/tensorrtx/yolov5/tracking/src/api/deepsort.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/tensorrtx/yolov5/tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/deepsort.dir/api/deepsort.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deepsort.dir/api/deepsort.cpp.o -c /home/nvidia/tensorrtx/yolov5/tracking/src/api/deepsort.cpp

CMakeFiles/deepsort.dir/api/deepsort.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deepsort.dir/api/deepsort.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/tensorrtx/yolov5/tracking/src/api/deepsort.cpp > CMakeFiles/deepsort.dir/api/deepsort.cpp.i

CMakeFiles/deepsort.dir/api/deepsort.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deepsort.dir/api/deepsort.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/tensorrtx/yolov5/tracking/src/api/deepsort.cpp -o CMakeFiles/deepsort.dir/api/deepsort.cpp.s

CMakeFiles/deepsort.dir/api/deepsort.cpp.o.requires:

.PHONY : CMakeFiles/deepsort.dir/api/deepsort.cpp.o.requires

CMakeFiles/deepsort.dir/api/deepsort.cpp.o.provides: CMakeFiles/deepsort.dir/api/deepsort.cpp.o.requires
	$(MAKE) -f CMakeFiles/deepsort.dir/build.make CMakeFiles/deepsort.dir/api/deepsort.cpp.o.provides.build
.PHONY : CMakeFiles/deepsort.dir/api/deepsort.cpp.o.provides

CMakeFiles/deepsort.dir/api/deepsort.cpp.o.provides.build: CMakeFiles/deepsort.dir/api/deepsort.cpp.o


# Object files for target deepsort
deepsort_OBJECTS = \
"CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.o" \
"CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.o" \
"CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.o" \
"CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.o" \
"CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.o" \
"CMakeFiles/deepsort.dir/feature/model.cpp.o" \
"CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.o" \
"CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.o" \
"CMakeFiles/deepsort.dir/matching/nn_matching.cpp.o" \
"CMakeFiles/deepsort.dir/matching/track.cpp.o" \
"CMakeFiles/deepsort.dir/matching/tracker.cpp.o" \
"CMakeFiles/deepsort.dir/api/deepsort.cpp.o"

# External object files for target deepsort
deepsort_EXTERNAL_OBJECTS =

libdeepsort.so: CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.o
libdeepsort.so: CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.o
libdeepsort.so: CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.o
libdeepsort.so: CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.o
libdeepsort.so: CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.o
libdeepsort.so: CMakeFiles/deepsort.dir/feature/model.cpp.o
libdeepsort.so: CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.o
libdeepsort.so: CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.o
libdeepsort.so: CMakeFiles/deepsort.dir/matching/nn_matching.cpp.o
libdeepsort.so: CMakeFiles/deepsort.dir/matching/track.cpp.o
libdeepsort.so: CMakeFiles/deepsort.dir/matching/tracker.cpp.o
libdeepsort.so: CMakeFiles/deepsort.dir/api/deepsort.cpp.o
libdeepsort.so: CMakeFiles/deepsort.dir/build.make
libdeepsort.so: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.1.1
libdeepsort.so: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.1.1
libdeepsort.so: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.1.1
libdeepsort.so: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.1.1
libdeepsort.so: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.1.1
libdeepsort.so: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.1.1
libdeepsort.so: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.1.1
libdeepsort.so: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.1.1
libdeepsort.so: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.1.1
libdeepsort.so: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.1.1
libdeepsort.so: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.1.1
libdeepsort.so: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.1.1
libdeepsort.so: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.1.1
libdeepsort.so: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.1.1
libdeepsort.so: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.1.1
libdeepsort.so: CMakeFiles/deepsort.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/tensorrtx/yolov5/tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Linking CXX shared library libdeepsort.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/deepsort.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/deepsort.dir/build: libdeepsort.so

.PHONY : CMakeFiles/deepsort.dir/build

CMakeFiles/deepsort.dir/requires: CMakeFiles/deepsort.dir/errmsg/errmsg.cpp.o.requires
CMakeFiles/deepsort.dir/requires: CMakeFiles/deepsort.dir/thirdPart/hungarianoper.cpp.o.requires
CMakeFiles/deepsort.dir/requires: CMakeFiles/deepsort.dir/thirdPart/munkres/munkres.cpp.o.requires
CMakeFiles/deepsort.dir/requires: CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/adapter.cpp.o.requires
CMakeFiles/deepsort.dir/requires: CMakeFiles/deepsort.dir/thirdPart/munkres/adapters/boostmatrixadapter.cpp.o.requires
CMakeFiles/deepsort.dir/requires: CMakeFiles/deepsort.dir/feature/model.cpp.o.requires
CMakeFiles/deepsort.dir/requires: CMakeFiles/deepsort.dir/matching/kalmanfilter.cpp.o.requires
CMakeFiles/deepsort.dir/requires: CMakeFiles/deepsort.dir/matching/linear_assignment.cpp.o.requires
CMakeFiles/deepsort.dir/requires: CMakeFiles/deepsort.dir/matching/nn_matching.cpp.o.requires
CMakeFiles/deepsort.dir/requires: CMakeFiles/deepsort.dir/matching/track.cpp.o.requires
CMakeFiles/deepsort.dir/requires: CMakeFiles/deepsort.dir/matching/tracker.cpp.o.requires
CMakeFiles/deepsort.dir/requires: CMakeFiles/deepsort.dir/api/deepsort.cpp.o.requires

.PHONY : CMakeFiles/deepsort.dir/requires

CMakeFiles/deepsort.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/deepsort.dir/cmake_clean.cmake
.PHONY : CMakeFiles/deepsort.dir/clean

CMakeFiles/deepsort.dir/depend:
	cd /home/nvidia/tensorrtx/yolov5/tracking/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/tensorrtx/yolov5/tracking/src /home/nvidia/tensorrtx/yolov5/tracking/src /home/nvidia/tensorrtx/yolov5/tracking/build /home/nvidia/tensorrtx/yolov5/tracking/build /home/nvidia/tensorrtx/yolov5/tracking/build/CMakeFiles/deepsort.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/deepsort.dir/depend

