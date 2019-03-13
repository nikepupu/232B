cmake_minimum_required(VERSION 3.8)
project(aog_lib)

set(CMAKE_CXX_STANDARD 11)

set( CMAKE_CXX_FLAGS " -g " )

find_package(Boost COMPONENTS filesystem log program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIR})

add_subdirectory(AOG_Base)
add_subdirectory(inference)
