add_executable(tf_c_test testC.cpp)
target_link_libraries(tf_c_test
        tfCDetectTools
        )

add_executable(tf_cxx_test testCXX.cpp)
target_link_libraries(tf_cxx_test
        tfCXXDetectTools
        )