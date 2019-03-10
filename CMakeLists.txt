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

add_executable(  res main.cpp detector.cpp saot_inference_Config.cpp filter.cpp misc.cpp)

target_link_libraries(res ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES})
