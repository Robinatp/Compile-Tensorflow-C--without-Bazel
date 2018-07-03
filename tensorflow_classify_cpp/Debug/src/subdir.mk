################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/classify.cpp \
../src/tensorflow_classify_cpp.cpp 

O_SRCS += \
../src/classify.o 

OBJS += \
./src/classify.o \
./src/tensorflow_classify_cpp.o 

CPP_DEPS += \
./src/classify.d \
./src/tensorflow_classify_cpp.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/home/ubuntu/eclipse-workspace-cpp/tensorflow_include/include -I/home/ubuntu/eclipse-workspace-cpp/tensorflow_include/include/third_party -O0 -std=c++11 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


