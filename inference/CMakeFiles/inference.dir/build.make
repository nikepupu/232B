# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/nikepupu/10D6122F101EC956/232B

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/nikepupu/10D6122F101EC956/232B

# Include any dependencies generated for this target.
include inference/CMakeFiles/inference.dir/depend.make

# Include the progress variables for this target.
include inference/CMakeFiles/inference.dir/progress.make

# Include the compile flags for this target's objects.
include inference/CMakeFiles/inference.dir/flags.make

inference/CMakeFiles/inference.dir/detector.cpp.o: inference/CMakeFiles/inference.dir/flags.make
inference/CMakeFiles/inference.dir/detector.cpp.o: inference/detector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/nikepupu/10D6122F101EC956/232B/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object inference/CMakeFiles/inference.dir/detector.cpp.o"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inference.dir/detector.cpp.o -c /media/nikepupu/10D6122F101EC956/232B/inference/detector.cpp

inference/CMakeFiles/inference.dir/detector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inference.dir/detector.cpp.i"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/nikepupu/10D6122F101EC956/232B/inference/detector.cpp > CMakeFiles/inference.dir/detector.cpp.i

inference/CMakeFiles/inference.dir/detector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inference.dir/detector.cpp.s"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/nikepupu/10D6122F101EC956/232B/inference/detector.cpp -o CMakeFiles/inference.dir/detector.cpp.s

inference/CMakeFiles/inference.dir/exponential_model.cpp.o: inference/CMakeFiles/inference.dir/flags.make
inference/CMakeFiles/inference.dir/exponential_model.cpp.o: inference/exponential_model.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/nikepupu/10D6122F101EC956/232B/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object inference/CMakeFiles/inference.dir/exponential_model.cpp.o"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inference.dir/exponential_model.cpp.o -c /media/nikepupu/10D6122F101EC956/232B/inference/exponential_model.cpp

inference/CMakeFiles/inference.dir/exponential_model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inference.dir/exponential_model.cpp.i"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/nikepupu/10D6122F101EC956/232B/inference/exponential_model.cpp > CMakeFiles/inference.dir/exponential_model.cpp.i

inference/CMakeFiles/inference.dir/exponential_model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inference.dir/exponential_model.cpp.s"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/nikepupu/10D6122F101EC956/232B/inference/exponential_model.cpp -o CMakeFiles/inference.dir/exponential_model.cpp.s

inference/CMakeFiles/inference.dir/filter.cpp.o: inference/CMakeFiles/inference.dir/flags.make
inference/CMakeFiles/inference.dir/filter.cpp.o: inference/filter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/nikepupu/10D6122F101EC956/232B/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object inference/CMakeFiles/inference.dir/filter.cpp.o"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inference.dir/filter.cpp.o -c /media/nikepupu/10D6122F101EC956/232B/inference/filter.cpp

inference/CMakeFiles/inference.dir/filter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inference.dir/filter.cpp.i"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/nikepupu/10D6122F101EC956/232B/inference/filter.cpp > CMakeFiles/inference.dir/filter.cpp.i

inference/CMakeFiles/inference.dir/filter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inference.dir/filter.cpp.s"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/nikepupu/10D6122F101EC956/232B/inference/filter.cpp -o CMakeFiles/inference.dir/filter.cpp.s

inference/CMakeFiles/inference.dir/layers/layer1.cpp.o: inference/CMakeFiles/inference.dir/flags.make
inference/CMakeFiles/inference.dir/layers/layer1.cpp.o: inference/layers/layer1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/nikepupu/10D6122F101EC956/232B/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object inference/CMakeFiles/inference.dir/layers/layer1.cpp.o"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inference.dir/layers/layer1.cpp.o -c /media/nikepupu/10D6122F101EC956/232B/inference/layers/layer1.cpp

inference/CMakeFiles/inference.dir/layers/layer1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inference.dir/layers/layer1.cpp.i"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/nikepupu/10D6122F101EC956/232B/inference/layers/layer1.cpp > CMakeFiles/inference.dir/layers/layer1.cpp.i

inference/CMakeFiles/inference.dir/layers/layer1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inference.dir/layers/layer1.cpp.s"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/nikepupu/10D6122F101EC956/232B/inference/layers/layer1.cpp -o CMakeFiles/inference.dir/layers/layer1.cpp.s

inference/CMakeFiles/inference.dir/layers/layer2.cpp.o: inference/CMakeFiles/inference.dir/flags.make
inference/CMakeFiles/inference.dir/layers/layer2.cpp.o: inference/layers/layer2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/nikepupu/10D6122F101EC956/232B/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object inference/CMakeFiles/inference.dir/layers/layer2.cpp.o"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inference.dir/layers/layer2.cpp.o -c /media/nikepupu/10D6122F101EC956/232B/inference/layers/layer2.cpp

inference/CMakeFiles/inference.dir/layers/layer2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inference.dir/layers/layer2.cpp.i"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/nikepupu/10D6122F101EC956/232B/inference/layers/layer2.cpp > CMakeFiles/inference.dir/layers/layer2.cpp.i

inference/CMakeFiles/inference.dir/layers/layer2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inference.dir/layers/layer2.cpp.s"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/nikepupu/10D6122F101EC956/232B/inference/layers/layer2.cpp -o CMakeFiles/inference.dir/layers/layer2.cpp.s

inference/CMakeFiles/inference.dir/layers/layer3.cpp.o: inference/CMakeFiles/inference.dir/flags.make
inference/CMakeFiles/inference.dir/layers/layer3.cpp.o: inference/layers/layer3.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/nikepupu/10D6122F101EC956/232B/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object inference/CMakeFiles/inference.dir/layers/layer3.cpp.o"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inference.dir/layers/layer3.cpp.o -c /media/nikepupu/10D6122F101EC956/232B/inference/layers/layer3.cpp

inference/CMakeFiles/inference.dir/layers/layer3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inference.dir/layers/layer3.cpp.i"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/nikepupu/10D6122F101EC956/232B/inference/layers/layer3.cpp > CMakeFiles/inference.dir/layers/layer3.cpp.i

inference/CMakeFiles/inference.dir/layers/layer3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inference.dir/layers/layer3.cpp.s"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/nikepupu/10D6122F101EC956/232B/inference/layers/layer3.cpp -o CMakeFiles/inference.dir/layers/layer3.cpp.s

inference/CMakeFiles/inference.dir/main.cpp.o: inference/CMakeFiles/inference.dir/flags.make
inference/CMakeFiles/inference.dir/main.cpp.o: inference/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/nikepupu/10D6122F101EC956/232B/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object inference/CMakeFiles/inference.dir/main.cpp.o"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inference.dir/main.cpp.o -c /media/nikepupu/10D6122F101EC956/232B/inference/main.cpp

inference/CMakeFiles/inference.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inference.dir/main.cpp.i"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/nikepupu/10D6122F101EC956/232B/inference/main.cpp > CMakeFiles/inference.dir/main.cpp.i

inference/CMakeFiles/inference.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inference.dir/main.cpp.s"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/nikepupu/10D6122F101EC956/232B/inference/main.cpp -o CMakeFiles/inference.dir/main.cpp.s

inference/CMakeFiles/inference.dir/misc.cpp.o: inference/CMakeFiles/inference.dir/flags.make
inference/CMakeFiles/inference.dir/misc.cpp.o: inference/misc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/nikepupu/10D6122F101EC956/232B/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object inference/CMakeFiles/inference.dir/misc.cpp.o"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inference.dir/misc.cpp.o -c /media/nikepupu/10D6122F101EC956/232B/inference/misc.cpp

inference/CMakeFiles/inference.dir/misc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inference.dir/misc.cpp.i"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/nikepupu/10D6122F101EC956/232B/inference/misc.cpp > CMakeFiles/inference.dir/misc.cpp.i

inference/CMakeFiles/inference.dir/misc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inference.dir/misc.cpp.s"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/nikepupu/10D6122F101EC956/232B/inference/misc.cpp -o CMakeFiles/inference.dir/misc.cpp.s

inference/CMakeFiles/inference.dir/saot_config.cpp.o: inference/CMakeFiles/inference.dir/flags.make
inference/CMakeFiles/inference.dir/saot_config.cpp.o: inference/saot_config.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/nikepupu/10D6122F101EC956/232B/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object inference/CMakeFiles/inference.dir/saot_config.cpp.o"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inference.dir/saot_config.cpp.o -c /media/nikepupu/10D6122F101EC956/232B/inference/saot_config.cpp

inference/CMakeFiles/inference.dir/saot_config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inference.dir/saot_config.cpp.i"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/nikepupu/10D6122F101EC956/232B/inference/saot_config.cpp > CMakeFiles/inference.dir/saot_config.cpp.i

inference/CMakeFiles/inference.dir/saot_config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inference.dir/saot_config.cpp.s"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/nikepupu/10D6122F101EC956/232B/inference/saot_config.cpp -o CMakeFiles/inference.dir/saot_config.cpp.s

inference/CMakeFiles/inference.dir/template.cpp.o: inference/CMakeFiles/inference.dir/flags.make
inference/CMakeFiles/inference.dir/template.cpp.o: inference/template.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/nikepupu/10D6122F101EC956/232B/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object inference/CMakeFiles/inference.dir/template.cpp.o"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inference.dir/template.cpp.o -c /media/nikepupu/10D6122F101EC956/232B/inference/template.cpp

inference/CMakeFiles/inference.dir/template.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inference.dir/template.cpp.i"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/nikepupu/10D6122F101EC956/232B/inference/template.cpp > CMakeFiles/inference.dir/template.cpp.i

inference/CMakeFiles/inference.dir/template.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inference.dir/template.cpp.s"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/nikepupu/10D6122F101EC956/232B/inference/template.cpp -o CMakeFiles/inference.dir/template.cpp.s

inference/CMakeFiles/inference.dir/util/file_util.cpp.o: inference/CMakeFiles/inference.dir/flags.make
inference/CMakeFiles/inference.dir/util/file_util.cpp.o: inference/util/file_util.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/nikepupu/10D6122F101EC956/232B/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object inference/CMakeFiles/inference.dir/util/file_util.cpp.o"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inference.dir/util/file_util.cpp.o -c /media/nikepupu/10D6122F101EC956/232B/inference/util/file_util.cpp

inference/CMakeFiles/inference.dir/util/file_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inference.dir/util/file_util.cpp.i"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/nikepupu/10D6122F101EC956/232B/inference/util/file_util.cpp > CMakeFiles/inference.dir/util/file_util.cpp.i

inference/CMakeFiles/inference.dir/util/file_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inference.dir/util/file_util.cpp.s"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/nikepupu/10D6122F101EC956/232B/inference/util/file_util.cpp -o CMakeFiles/inference.dir/util/file_util.cpp.s

inference/CMakeFiles/inference.dir/util/mat_util.cpp.o: inference/CMakeFiles/inference.dir/flags.make
inference/CMakeFiles/inference.dir/util/mat_util.cpp.o: inference/util/mat_util.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/nikepupu/10D6122F101EC956/232B/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object inference/CMakeFiles/inference.dir/util/mat_util.cpp.o"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inference.dir/util/mat_util.cpp.o -c /media/nikepupu/10D6122F101EC956/232B/inference/util/mat_util.cpp

inference/CMakeFiles/inference.dir/util/mat_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inference.dir/util/mat_util.cpp.i"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/nikepupu/10D6122F101EC956/232B/inference/util/mat_util.cpp > CMakeFiles/inference.dir/util/mat_util.cpp.i

inference/CMakeFiles/inference.dir/util/mat_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inference.dir/util/mat_util.cpp.s"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/nikepupu/10D6122F101EC956/232B/inference/util/mat_util.cpp -o CMakeFiles/inference.dir/util/mat_util.cpp.s

inference/CMakeFiles/inference.dir/util/vis_util.cpp.o: inference/CMakeFiles/inference.dir/flags.make
inference/CMakeFiles/inference.dir/util/vis_util.cpp.o: inference/util/vis_util.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/nikepupu/10D6122F101EC956/232B/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object inference/CMakeFiles/inference.dir/util/vis_util.cpp.o"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/inference.dir/util/vis_util.cpp.o -c /media/nikepupu/10D6122F101EC956/232B/inference/util/vis_util.cpp

inference/CMakeFiles/inference.dir/util/vis_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inference.dir/util/vis_util.cpp.i"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/nikepupu/10D6122F101EC956/232B/inference/util/vis_util.cpp > CMakeFiles/inference.dir/util/vis_util.cpp.i

inference/CMakeFiles/inference.dir/util/vis_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inference.dir/util/vis_util.cpp.s"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/nikepupu/10D6122F101EC956/232B/inference/util/vis_util.cpp -o CMakeFiles/inference.dir/util/vis_util.cpp.s

# Object files for target inference
inference_OBJECTS = \
"CMakeFiles/inference.dir/detector.cpp.o" \
"CMakeFiles/inference.dir/exponential_model.cpp.o" \
"CMakeFiles/inference.dir/filter.cpp.o" \
"CMakeFiles/inference.dir/layers/layer1.cpp.o" \
"CMakeFiles/inference.dir/layers/layer2.cpp.o" \
"CMakeFiles/inference.dir/layers/layer3.cpp.o" \
"CMakeFiles/inference.dir/main.cpp.o" \
"CMakeFiles/inference.dir/misc.cpp.o" \
"CMakeFiles/inference.dir/saot_config.cpp.o" \
"CMakeFiles/inference.dir/template.cpp.o" \
"CMakeFiles/inference.dir/util/file_util.cpp.o" \
"CMakeFiles/inference.dir/util/mat_util.cpp.o" \
"CMakeFiles/inference.dir/util/vis_util.cpp.o"

# External object files for target inference
inference_EXTERNAL_OBJECTS =

inference/inference: inference/CMakeFiles/inference.dir/detector.cpp.o
inference/inference: inference/CMakeFiles/inference.dir/exponential_model.cpp.o
inference/inference: inference/CMakeFiles/inference.dir/filter.cpp.o
inference/inference: inference/CMakeFiles/inference.dir/layers/layer1.cpp.o
inference/inference: inference/CMakeFiles/inference.dir/layers/layer2.cpp.o
inference/inference: inference/CMakeFiles/inference.dir/layers/layer3.cpp.o
inference/inference: inference/CMakeFiles/inference.dir/main.cpp.o
inference/inference: inference/CMakeFiles/inference.dir/misc.cpp.o
inference/inference: inference/CMakeFiles/inference.dir/saot_config.cpp.o
inference/inference: inference/CMakeFiles/inference.dir/template.cpp.o
inference/inference: inference/CMakeFiles/inference.dir/util/file_util.cpp.o
inference/inference: inference/CMakeFiles/inference.dir/util/mat_util.cpp.o
inference/inference: inference/CMakeFiles/inference.dir/util/vis_util.cpp.o
inference/inference: inference/CMakeFiles/inference.dir/build.make
inference/inference: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
inference/inference: /usr/lib/x86_64-linux-gnu/libboost_log.so
inference/inference: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
inference/inference: /usr/lib/x86_64-linux-gnu/libboost_system.so
inference/inference: /usr/lib/x86_64-linux-gnu/libboost_log_setup.so
inference/inference: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
inference/inference: /usr/lib/x86_64-linux-gnu/libboost_thread.so
inference/inference: /usr/lib/x86_64-linux-gnu/libboost_regex.so
inference/inference: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
inference/inference: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libboost_regex.so
inference/inference: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
inference/inference: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.9
inference/inference: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.9
inference/inference: inference/CMakeFiles/inference.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/nikepupu/10D6122F101EC956/232B/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Linking CXX executable inference"
	cd /media/nikepupu/10D6122F101EC956/232B/inference && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/inference.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
inference/CMakeFiles/inference.dir/build: inference/inference

.PHONY : inference/CMakeFiles/inference.dir/build

inference/CMakeFiles/inference.dir/clean:
	cd /media/nikepupu/10D6122F101EC956/232B/inference && $(CMAKE_COMMAND) -P CMakeFiles/inference.dir/cmake_clean.cmake
.PHONY : inference/CMakeFiles/inference.dir/clean

inference/CMakeFiles/inference.dir/depend:
	cd /media/nikepupu/10D6122F101EC956/232B && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/nikepupu/10D6122F101EC956/232B /media/nikepupu/10D6122F101EC956/232B/inference /media/nikepupu/10D6122F101EC956/232B /media/nikepupu/10D6122F101EC956/232B/inference /media/nikepupu/10D6122F101EC956/232B/inference/CMakeFiles/inference.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : inference/CMakeFiles/inference.dir/depend

