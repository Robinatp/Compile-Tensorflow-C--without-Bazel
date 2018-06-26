CC = g++
CFLAGS = -std=c++11 -g -Wall -D_DEBUG -Wshadow -Wno-sign-compare -w
INC = -I/usr/local/include/eigen3
INC += -I../include/third_party
INC += -I../include
INC += -I../include/nsync/public/
LDFLAGS =  -lprotobuf -pthread -lpthread
LDFLAGS += -ltensorflow_cc -ltensorflow_framework

all: main

main:
	$(CC) $(CFLAGS) -o main tf_example.cpp $(INC) $(LDFLAGS)
run:
	./main --image=../data/grace_hopper.jpg --graph=../data/inception_v3_2016_08_28_frozen.pb --labels=../data/imagenet_slim_labels.txt
clean:
	rm -f main

