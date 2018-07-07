set(common_souce Detection.cpp Common.cpp)

add_library(tfCXXDetectTools SHARED
        CXXTFDetector.cpp
        ${common_souce})
target_link_libraries(tfCXXDetectTools
        tensorflow_cc
        ${OpenCV_LIBS}
        )

add_library(tfCDetectTools SHARED
        CTFDetector.cpp
        ${common_souce})
target_link_libraries(tfCDetectTools
        tensorflow_cc
        ${OpenCV_LIBS}
        )