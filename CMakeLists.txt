cmake_minimum_required(VERSION 2.8)
ADD_DEFINITIONS(-DBOOST_LOG_DYN_LINK)
set(OpenCV_DIR /installation/OpenCV-/share/OpenCV)
find_package( OpenCV REQUIRED )
set(REQUIRED_BOOST_LIBRARIES)
list(APPEND REQUIRED_BOOST_LIBRARIES "log_setup" "log" "serialization" "system" "filesystem" "date_time")
set(Boost_USE_MULTITHREADED ON)
find_package(Boost
COMPONENTS "${REQUIRED_BOOST_LIBRARIES}"
REQUIRED
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g")
file(GLOB SAOT_HEADER_FILES 
          ${PROJECT_SOURCE_DIR}/*.hpp
          ${PROJECT_SOURCE_DIR}/layers/*.hpp
          ${PROJECT_SOURCE_DIR}/util/*.hpp
)

file(GLOB SAOT_SOURCE_FILES 
          ${PROJECT_SOURCE_DIR}/*.cpp
          ${PROJECT_SOURCE_DIR}/layers/*.cpp
          ${PROJECT_SOURCE_DIR}/util/*.cpp
)

add_executable(res ${SAOT_HEADER_FILES} ${SAOT_SOURCE_FILES})

target_link_libraries(res ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES})
