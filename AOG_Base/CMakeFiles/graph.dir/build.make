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
CMAKE_SOURCE_DIR = /home/nikepupu/Desktop/STAT232B_AOGLib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nikepupu/Desktop/STAT232B_AOGLib

# Include any dependencies generated for this target.
include AOG_Base/CMakeFiles/graph.dir/depend.make

# Include the progress variables for this target.
include AOG_Base/CMakeFiles/graph.dir/progress.make

# Include the compile flags for this target's objects.
include AOG_Base/CMakeFiles/graph.dir/flags.make

AOG_Base/CMakeFiles/graph.dir/grammar_example.cpp.o: AOG_Base/CMakeFiles/graph.dir/flags.make
AOG_Base/CMakeFiles/graph.dir/grammar_example.cpp.o: AOG_Base/grammar_example.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nikepupu/Desktop/STAT232B_AOGLib/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object AOG_Base/CMakeFiles/graph.dir/grammar_example.cpp.o"
	cd /home/nikepupu/Desktop/STAT232B_AOGLib/AOG_Base && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/graph.dir/grammar_example.cpp.o -c /home/nikepupu/Desktop/STAT232B_AOGLib/AOG_Base/grammar_example.cpp

AOG_Base/CMakeFiles/graph.dir/grammar_example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/graph.dir/grammar_example.cpp.i"
	cd /home/nikepupu/Desktop/STAT232B_AOGLib/AOG_Base && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nikepupu/Desktop/STAT232B_AOGLib/AOG_Base/grammar_example.cpp > CMakeFiles/graph.dir/grammar_example.cpp.i

AOG_Base/CMakeFiles/graph.dir/grammar_example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/graph.dir/grammar_example.cpp.s"
	cd /home/nikepupu/Desktop/STAT232B_AOGLib/AOG_Base && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nikepupu/Desktop/STAT232B_AOGLib/AOG_Base/grammar_example.cpp -o CMakeFiles/graph.dir/grammar_example.cpp.s

# Object files for target graph
graph_OBJECTS = \
"CMakeFiles/graph.dir/grammar_example.cpp.o"

# External object files for target graph
graph_EXTERNAL_OBJECTS =

AOG_Base/graph: AOG_Base/CMakeFiles/graph.dir/grammar_example.cpp.o
AOG_Base/graph: AOG_Base/CMakeFiles/graph.dir/build.make
AOG_Base/graph: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
AOG_Base/graph: /usr/lib/x86_64-linux-gnu/libboost_log.so
AOG_Base/graph: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
AOG_Base/graph: /usr/lib/x86_64-linux-gnu/libboost_system.so
AOG_Base/graph: /usr/lib/x86_64-linux-gnu/libboost_log_setup.so
AOG_Base/graph: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
AOG_Base/graph: /usr/lib/x86_64-linux-gnu/libboost_thread.so
AOG_Base/graph: /usr/lib/x86_64-linux-gnu/libboost_regex.so
AOG_Base/graph: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
AOG_Base/graph: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
AOG_Base/graph: AOG_Base/CMakeFiles/graph.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nikepupu/Desktop/STAT232B_AOGLib/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable graph"
	cd /home/nikepupu/Desktop/STAT232B_AOGLib/AOG_Base && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/graph.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
AOG_Base/CMakeFiles/graph.dir/build: AOG_Base/graph

.PHONY : AOG_Base/CMakeFiles/graph.dir/build

AOG_Base/CMakeFiles/graph.dir/clean:
	cd /home/nikepupu/Desktop/STAT232B_AOGLib/AOG_Base && $(CMAKE_COMMAND) -P CMakeFiles/graph.dir/cmake_clean.cmake
.PHONY : AOG_Base/CMakeFiles/graph.dir/clean

AOG_Base/CMakeFiles/graph.dir/depend:
	cd /home/nikepupu/Desktop/STAT232B_AOGLib && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nikepupu/Desktop/STAT232B_AOGLib /home/nikepupu/Desktop/STAT232B_AOGLib/AOG_Base /home/nikepupu/Desktop/STAT232B_AOGLib /home/nikepupu/Desktop/STAT232B_AOGLib/AOG_Base /home/nikepupu/Desktop/STAT232B_AOGLib/AOG_Base/CMakeFiles/graph.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : AOG_Base/CMakeFiles/graph.dir/depend
