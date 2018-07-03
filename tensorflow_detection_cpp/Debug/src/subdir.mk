################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/object_detection.cpp \
../src/tensorflow_detection_cpp.cpp 

OBJS += \
./src/object_detection.o \
./src/tensorflow_detection_cpp.o 

CPP_DEPS += \
./src/object_detection.d \
./src/tensorflow_detection_cpp.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/home/ubuntu/eclipse-workspace-cpp/tensorflow_include/include -I/home/ubuntu/eclipse-workspace-cpp/tensorflow_include/include/third_party -O0 -std=c++11 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


