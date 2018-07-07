cmake_minimum_required(VERSION 2.8)
project(tf_c_test)

option(USE_GPU "" OFF)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(OpenCV REQUIRED)

include_directories(#you must path to you include
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/thirdParty/include
        ${PROJECT_SOURCE_DIR}/thirdParty/include/third_party)
if (USE_GPU)
    link_directories(${PROJECT_SOURCE_DIR}/thirdParty/lib/GPU)
else(USE_GPU)
    link_directories(${PROJECT_SOURCE_DIR}/thirdParty/lib)
endif (USE_GPU)

add_subdirectory(src)
add_subdirectory(tests)