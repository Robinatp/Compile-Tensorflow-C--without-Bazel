cmake_minimum_required (VERSION 2.8.8)
project (tf_example)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -W")

#source file define
aux_source_directory(./src DIR_SRCS)

# ==================== PATHS TO SPECIFY! ==================== #
# TensorFlow lib directory
link_directories(./lib)
# TensorFlow headers
include_directories(
  ./include
  ./include/third_party/
  )
  
# Link TensorFlow libs
target_link_libraries(tf_example tensorflow tensorflow_cc tensorflow_framework)

add_executable(tf_example ${DIR_SRCS} ${HEADERS_FILES})
