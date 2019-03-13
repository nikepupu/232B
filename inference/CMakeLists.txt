find_package(OpenCV REQUIRED)

ADD_DEFINITIONS(-DBOOST_LOG_DYN_LINK)
file(GLOB SAOT_HEADER_FILES 
          ${PROJECT_SOURCE_DIR}/inference/*.hpp
          ${PROJECT_SOURCE_DIR}/inference/layers/*.hpp
          ${PROJECT_SOURCE_DIR}/inference/util/*.hpp
)

file(GLOB SAOT_SOURCE_FILES 
          ${PROJECT_SOURCE_DIR}/inference/*.cpp
          ${PROJECT_SOURCE_DIR}/inference/layers/*.cpp
          ${PROJECT_SOURCE_DIR}/inference/util/*.cpp
)

add_executable(inference ${SAOT_HEADER_FILES} ${SAOT_SOURCE_FILES})

target_link_libraries(inference 
    ${Boost_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${OpenCV_LIBS}
)
