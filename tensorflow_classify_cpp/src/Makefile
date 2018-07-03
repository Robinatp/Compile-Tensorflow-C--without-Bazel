#######################
# Makefile
#######################

# compile and lib parameter
CC      := g++

LIBS    :=

LDFLAGS :=  -lprotobuf -lpthread -lpthread
LDFLAGS += -ltensorflow_cc 

DEFINES :=

MY_DIR  := /home/ubuntu/eclipse-workspace/facenet/tensorflow_cpp
INCLUDE := -I/usr/local/include/eigen3
INCLUDE += -I$(MY_DIR)/include/third_party
INCLUDE += -I$(MY_DIR)/include
INCLUDE += -I$(MY_DIR)/include/nsync/public/

CFLAGS = -std=c++11 -g -Wall -D_DEBUG -Wshadow -Wno-sign-compare -w
CXXFLAGS:= 

# link parameter
LIB := libclassify.so


#link
$(LIB):classify.o
	$(CC) -shared -o -fPIC -o $@ $^  $(LDFLAGS)
#	sudo cp $@ /usr/local/lib
#	sudo ldconfig
#compile
classify.o:classify.cpp
	$(CC) -c -fPIC $^ $(CFLAGS) -o $@ $(INCLUDE)

# clean
clean:
	 rm -fr *.o
	 rm -fr *.so
    

