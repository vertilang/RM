# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vertilang/code/RM_practice

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vertilang/code/RM_practice/build

# Include any dependencies generated for this target.
include CMakeFiles/RM_prictice.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/RM_prictice.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/RM_prictice.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/RM_prictice.dir/flags.make

CMakeFiles/RM_prictice.dir/autoaim/RM_prictice_generated_preprocess.cu.o: /home/vertilang/code/RM_practice/autoaim/preprocess.cu
CMakeFiles/RM_prictice.dir/autoaim/RM_prictice_generated_preprocess.cu.o: CMakeFiles/RM_prictice.dir/autoaim/RM_prictice_generated_preprocess.cu.o.depend
CMakeFiles/RM_prictice.dir/autoaim/RM_prictice_generated_preprocess.cu.o: CMakeFiles/RM_prictice.dir/autoaim/RM_prictice_generated_preprocess.cu.o.Release.cmake
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/vertilang/code/RM_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/RM_prictice.dir/autoaim/RM_prictice_generated_preprocess.cu.o"
	cd /home/vertilang/code/RM_practice/build/CMakeFiles/RM_prictice.dir/autoaim && /usr/local/bin/cmake -E make_directory /home/vertilang/code/RM_practice/build/CMakeFiles/RM_prictice.dir/./autoaim/.
	cd /home/vertilang/code/RM_practice/build/CMakeFiles/RM_prictice.dir/autoaim && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Release -D generated_file:STRING=/home/vertilang/code/RM_practice/build/CMakeFiles/RM_prictice.dir/./autoaim/./RM_prictice_generated_preprocess.cu.o -D generated_cubin_file:STRING=/home/vertilang/code/RM_practice/build/CMakeFiles/RM_prictice.dir/./autoaim/./RM_prictice_generated_preprocess.cu.o.cubin.txt -P /home/vertilang/code/RM_practice/build/CMakeFiles/RM_prictice.dir/./autoaim/RM_prictice_generated_preprocess.cu.o.Release.cmake

CMakeFiles/RM_prictice.dir/autoaim/TRTModule.cpp.o: CMakeFiles/RM_prictice.dir/flags.make
CMakeFiles/RM_prictice.dir/autoaim/TRTModule.cpp.o: /home/vertilang/code/RM_practice/autoaim/TRTModule.cpp
CMakeFiles/RM_prictice.dir/autoaim/TRTModule.cpp.o: CMakeFiles/RM_prictice.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vertilang/code/RM_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/RM_prictice.dir/autoaim/TRTModule.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RM_prictice.dir/autoaim/TRTModule.cpp.o -MF CMakeFiles/RM_prictice.dir/autoaim/TRTModule.cpp.o.d -o CMakeFiles/RM_prictice.dir/autoaim/TRTModule.cpp.o -c /home/vertilang/code/RM_practice/autoaim/TRTModule.cpp

CMakeFiles/RM_prictice.dir/autoaim/TRTModule.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/RM_prictice.dir/autoaim/TRTModule.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vertilang/code/RM_practice/autoaim/TRTModule.cpp > CMakeFiles/RM_prictice.dir/autoaim/TRTModule.cpp.i

CMakeFiles/RM_prictice.dir/autoaim/TRTModule.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/RM_prictice.dir/autoaim/TRTModule.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vertilang/code/RM_practice/autoaim/TRTModule.cpp -o CMakeFiles/RM_prictice.dir/autoaim/TRTModule.cpp.s

CMakeFiles/RM_prictice.dir/DaHeng/DaHengCamera.cpp.o: CMakeFiles/RM_prictice.dir/flags.make
CMakeFiles/RM_prictice.dir/DaHeng/DaHengCamera.cpp.o: /home/vertilang/code/RM_practice/DaHeng/DaHengCamera.cpp
CMakeFiles/RM_prictice.dir/DaHeng/DaHengCamera.cpp.o: CMakeFiles/RM_prictice.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vertilang/code/RM_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/RM_prictice.dir/DaHeng/DaHengCamera.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RM_prictice.dir/DaHeng/DaHengCamera.cpp.o -MF CMakeFiles/RM_prictice.dir/DaHeng/DaHengCamera.cpp.o.d -o CMakeFiles/RM_prictice.dir/DaHeng/DaHengCamera.cpp.o -c /home/vertilang/code/RM_practice/DaHeng/DaHengCamera.cpp

CMakeFiles/RM_prictice.dir/DaHeng/DaHengCamera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/RM_prictice.dir/DaHeng/DaHengCamera.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vertilang/code/RM_practice/DaHeng/DaHengCamera.cpp > CMakeFiles/RM_prictice.dir/DaHeng/DaHengCamera.cpp.i

CMakeFiles/RM_prictice.dir/DaHeng/DaHengCamera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/RM_prictice.dir/DaHeng/DaHengCamera.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vertilang/code/RM_practice/DaHeng/DaHengCamera.cpp -o CMakeFiles/RM_prictice.dir/DaHeng/DaHengCamera.cpp.s

CMakeFiles/RM_prictice.dir/src/Send_Receive.cpp.o: CMakeFiles/RM_prictice.dir/flags.make
CMakeFiles/RM_prictice.dir/src/Send_Receive.cpp.o: /home/vertilang/code/RM_practice/src/Send_Receive.cpp
CMakeFiles/RM_prictice.dir/src/Send_Receive.cpp.o: CMakeFiles/RM_prictice.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vertilang/code/RM_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/RM_prictice.dir/src/Send_Receive.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RM_prictice.dir/src/Send_Receive.cpp.o -MF CMakeFiles/RM_prictice.dir/src/Send_Receive.cpp.o.d -o CMakeFiles/RM_prictice.dir/src/Send_Receive.cpp.o -c /home/vertilang/code/RM_practice/src/Send_Receive.cpp

CMakeFiles/RM_prictice.dir/src/Send_Receive.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/RM_prictice.dir/src/Send_Receive.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vertilang/code/RM_practice/src/Send_Receive.cpp > CMakeFiles/RM_prictice.dir/src/Send_Receive.cpp.i

CMakeFiles/RM_prictice.dir/src/Send_Receive.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/RM_prictice.dir/src/Send_Receive.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vertilang/code/RM_practice/src/Send_Receive.cpp -o CMakeFiles/RM_prictice.dir/src/Send_Receive.cpp.s

CMakeFiles/RM_prictice.dir/src/predict.cpp.o: CMakeFiles/RM_prictice.dir/flags.make
CMakeFiles/RM_prictice.dir/src/predict.cpp.o: /home/vertilang/code/RM_practice/src/predict.cpp
CMakeFiles/RM_prictice.dir/src/predict.cpp.o: CMakeFiles/RM_prictice.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vertilang/code/RM_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/RM_prictice.dir/src/predict.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RM_prictice.dir/src/predict.cpp.o -MF CMakeFiles/RM_prictice.dir/src/predict.cpp.o.d -o CMakeFiles/RM_prictice.dir/src/predict.cpp.o -c /home/vertilang/code/RM_practice/src/predict.cpp

CMakeFiles/RM_prictice.dir/src/predict.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/RM_prictice.dir/src/predict.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vertilang/code/RM_practice/src/predict.cpp > CMakeFiles/RM_prictice.dir/src/predict.cpp.i

CMakeFiles/RM_prictice.dir/src/predict.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/RM_prictice.dir/src/predict.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vertilang/code/RM_practice/src/predict.cpp -o CMakeFiles/RM_prictice.dir/src/predict.cpp.s

CMakeFiles/RM_prictice.dir/src/thread.cpp.o: CMakeFiles/RM_prictice.dir/flags.make
CMakeFiles/RM_prictice.dir/src/thread.cpp.o: /home/vertilang/code/RM_practice/src/thread.cpp
CMakeFiles/RM_prictice.dir/src/thread.cpp.o: CMakeFiles/RM_prictice.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vertilang/code/RM_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/RM_prictice.dir/src/thread.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RM_prictice.dir/src/thread.cpp.o -MF CMakeFiles/RM_prictice.dir/src/thread.cpp.o.d -o CMakeFiles/RM_prictice.dir/src/thread.cpp.o -c /home/vertilang/code/RM_practice/src/thread.cpp

CMakeFiles/RM_prictice.dir/src/thread.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/RM_prictice.dir/src/thread.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vertilang/code/RM_practice/src/thread.cpp > CMakeFiles/RM_prictice.dir/src/thread.cpp.i

CMakeFiles/RM_prictice.dir/src/thread.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/RM_prictice.dir/src/thread.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vertilang/code/RM_practice/src/thread.cpp -o CMakeFiles/RM_prictice.dir/src/thread.cpp.s

CMakeFiles/RM_prictice.dir/main.cpp.o: CMakeFiles/RM_prictice.dir/flags.make
CMakeFiles/RM_prictice.dir/main.cpp.o: /home/vertilang/code/RM_practice/main.cpp
CMakeFiles/RM_prictice.dir/main.cpp.o: CMakeFiles/RM_prictice.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vertilang/code/RM_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/RM_prictice.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/RM_prictice.dir/main.cpp.o -MF CMakeFiles/RM_prictice.dir/main.cpp.o.d -o CMakeFiles/RM_prictice.dir/main.cpp.o -c /home/vertilang/code/RM_practice/main.cpp

CMakeFiles/RM_prictice.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/RM_prictice.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vertilang/code/RM_practice/main.cpp > CMakeFiles/RM_prictice.dir/main.cpp.i

CMakeFiles/RM_prictice.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/RM_prictice.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vertilang/code/RM_practice/main.cpp -o CMakeFiles/RM_prictice.dir/main.cpp.s

# Object files for target RM_prictice
RM_prictice_OBJECTS = \
"CMakeFiles/RM_prictice.dir/autoaim/TRTModule.cpp.o" \
"CMakeFiles/RM_prictice.dir/DaHeng/DaHengCamera.cpp.o" \
"CMakeFiles/RM_prictice.dir/src/Send_Receive.cpp.o" \
"CMakeFiles/RM_prictice.dir/src/predict.cpp.o" \
"CMakeFiles/RM_prictice.dir/src/thread.cpp.o" \
"CMakeFiles/RM_prictice.dir/main.cpp.o"

# External object files for target RM_prictice
RM_prictice_EXTERNAL_OBJECTS = \
"/home/vertilang/code/RM_practice/build/CMakeFiles/RM_prictice.dir/autoaim/RM_prictice_generated_preprocess.cu.o"

RM_prictice: CMakeFiles/RM_prictice.dir/autoaim/TRTModule.cpp.o
RM_prictice: CMakeFiles/RM_prictice.dir/DaHeng/DaHengCamera.cpp.o
RM_prictice: CMakeFiles/RM_prictice.dir/src/Send_Receive.cpp.o
RM_prictice: CMakeFiles/RM_prictice.dir/src/predict.cpp.o
RM_prictice: CMakeFiles/RM_prictice.dir/src/thread.cpp.o
RM_prictice: CMakeFiles/RM_prictice.dir/main.cpp.o
RM_prictice: CMakeFiles/RM_prictice.dir/autoaim/RM_prictice_generated_preprocess.cu.o
RM_prictice: CMakeFiles/RM_prictice.dir/build.make
RM_prictice: /usr/local/cuda/lib64/libcudart.so
RM_prictice: /usr/local/lib/libopencv_gapi.so.4.5.5
RM_prictice: /usr/local/lib/libopencv_highgui.so.4.5.5
RM_prictice: /usr/local/lib/libopencv_ml.so.4.5.5
RM_prictice: /usr/local/lib/libopencv_objdetect.so.4.5.5
RM_prictice: /usr/local/lib/libopencv_photo.so.4.5.5
RM_prictice: /usr/local/lib/libopencv_stitching.so.4.5.5
RM_prictice: /usr/local/lib/libopencv_video.so.4.5.5
RM_prictice: /usr/local/lib/libopencv_videoio.so.4.5.5
RM_prictice: /home/vertilang/Galaxy_Linux-x86_Gige-U3_32bits-64bits_1.5.2303.9221/Galaxy_camera/lib/x86_64/libgxiapi.so
RM_prictice: /usr/local/lib/libfmt.a
RM_prictice: /usr/local/lib/libopencv_imgcodecs.so.4.5.5
RM_prictice: /usr/local/lib/libopencv_dnn.so.4.5.5
RM_prictice: /usr/local/lib/libopencv_calib3d.so.4.5.5
RM_prictice: /usr/local/lib/libopencv_features2d.so.4.5.5
RM_prictice: /usr/local/lib/libopencv_flann.so.4.5.5
RM_prictice: /usr/local/lib/libopencv_imgproc.so.4.5.5
RM_prictice: /usr/local/lib/libopencv_core.so.4.5.5
RM_prictice: CMakeFiles/RM_prictice.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/vertilang/code/RM_practice/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable RM_prictice"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RM_prictice.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/RM_prictice.dir/build: RM_prictice
.PHONY : CMakeFiles/RM_prictice.dir/build

CMakeFiles/RM_prictice.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/RM_prictice.dir/cmake_clean.cmake
.PHONY : CMakeFiles/RM_prictice.dir/clean

CMakeFiles/RM_prictice.dir/depend: CMakeFiles/RM_prictice.dir/autoaim/RM_prictice_generated_preprocess.cu.o
	cd /home/vertilang/code/RM_practice/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vertilang/code/RM_practice /home/vertilang/code/RM_practice /home/vertilang/code/RM_practice/build /home/vertilang/code/RM_practice/build /home/vertilang/code/RM_practice/build/CMakeFiles/RM_prictice.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/RM_prictice.dir/depend

