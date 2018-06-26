cmake_minimum_required (VERSION 2.8.8)
project (tf_example)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -W")
#file(GLOB HEADERS_FILES include/*.h)
#file(GLOB SRC_FILES src/*.cpp)

aux_source_directory(./src DIR_SRCS)
link_directories(./lib)
include_directories(
  ./include/third_party/
  ./include
  )

add_executable(tf_example ${DIR_SRCS} ${HEADERS_FILES})
target_link_libraries(tf_example tensorflow tensorflow_cc tensorflow_framework)
