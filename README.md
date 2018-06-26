# Compile-Tensorflow-C--without-Bazel
Compile Tensorflow C++ without Bazel, You could create a C++ Tensorflow project in your favorite C++ IDEs and build it with Makefile or CMake and you will need to do some extra work to allow gcc to be able to compile successfully C++ Tensorflow codes. 

#This work is inspired by [tuanphuc](https://tuanphuc.github.io/standalone-tensorflow-cpp/),Thanks for his blog!




Compile Tensorflow C++ without Bazel

Updates:

The configurations to compile for tensorflow 1.6.0 is described in this post. If you want tensorflow to work nicely with OpenCV, follow that post.
In this post, I will give detailed instructions on how to compile the official C++ Tensorflow project label_image with gcc instead of bazel.

The reason why I write this blog is because officially, to compile a C++ Tensorflow project, you have to integrate it in the source tree of tensorflow, create a BUILD file and compile it with bazel. For some reason, if you want to create a C++ Tensorflow project in your favorite C++ IDEs and build it with Makefile or CMake, you will need to do some extra work to allow gcc to be able to compile successfully C++ Tensorflow codes. The detailed instructions are in the second part, you can skip the first part (Create a Ubuntu docker image) if you want to do directly on your machine instead of on a docker image.

I. Create a Ubuntu docker image

In the first part, I will create a docker image with latest version of tensorflow on the latest stable Ubuntu:

Ubuntu 17.10
gcc 7.2 (comes with Ubuntu 17.10)
tensorflow 1.4.0
The only reason why I use Docker is to create an independent environment to test the latest version of tensorfow on the latest version of Ubuntu.

I assume that you know the basic of Docker, here I create a ubuntu 17.10 image with this Dockerfile:

FROM ubuntu:17.10

# Install.
RUN \
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y build-essential && \
  apt-get install -y software-properties-common && \
  apt-get install -y byobu curl git htop man unzip vim wget && \
  apt-get update && \
  apt-get install python-dev python-pip python-setuptools python-sphinx python-yaml python-h5py python3-pip python-numpy python-scipy python-nose && \
  rm -rf /var/lib/apt/lists/*
Then use the following command to create the image:

docker build -t ubuntu:17.10 .
From now on, we will work on the interactive shell of docker image, to go into the shell:

docker run -it ubuntu:17.10 /bin/sh
II. Steps to make a standalone C++ Tensorflow

1. Compile tensorflow

Install bazel to compile tensorflow:

apt-get install openjdk-8-jdk
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
Clone tensorflow repo:

git clone https://github.com/tensorflow/tensorflow.git
Compile tensorflow for c++ (default parameters, python3)

cd tensorflow
./configure
bazel build //tensorflow:libtensorflow_cc.so
2. Install additional libraries

You need to install cmake to compile others libraries, here I install cmake 3.9.6:

wget https://cmake.org/files/v3.9/cmake-3.9.6.tar.gz
tar -xzvf cmake-3.9.6.tar.gz
cd cmake-3.9.6
./configure
make
make install
Install protobuf from sources:

cd /home
apt-get install autoconf automake libtool
git clone https://github.com/google/protobuf.git
git checkout v3.4.0
cd protobuf
./autogen.sh
./configure
make
make check
make install

#or I strongly suggest use the script as bellows:
export PROTOBUF_DIR=/workspace/software/tensorflow/tensorflow/contrib/makefile/
cd PROTOBUF_DIR
sudo ./build_all_linux.sh
cd -
sudo cp -r PROTOBUF_DIR/gen/protobuf/include/google third_party/

ldconfig

Install Eigen 3.3.4 from sources

cd /home
wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2
tar -xzvf 3.3.4.tar.bz2
cd eigen-folder
mkdir build
cd build
cmake ..
make
make install
3. Create and compile a standalone tensorflow C++ project

Create a standalone folder to test compilation of C++ tensorflow with gcc

cd /home
mkdir standalone
To compile tensorflow with gcc, it has to get all the header files required to compile. Create an include folder and copy header files to that folder:

export TENSORFLOW_DIR=/home/tensorflow
cd /home/standalone
mkdir -p include/third_party
cp -r $TENSORFLOW_DIR/tensorflow include/third_party/
cp -r $TENSORFLOW_DIR/bazel-genfiles/tensorflow include/third_party/
cp -r $TENSORFLOW_DIR/third_party/eigen3 include/third_party/
cp -r /home/protobuf/src/google include/third_party/
cp -r /home/eigen-folder/. include/third_party/eigen3/
cp -r include/third_party/eigen3/Eigen include/third_party/
Clone google nsync:

cd /home/standalone/include
git clone https://github.com/google/nsync.git
We need also 2 libraries libtensorflow_cc.so and libtensorflow_framework.so. Copy those libraries to /usr/local/lib:

cd /home/standalone
mkdir lib
cp -r $TENSORFLOW_DIR/bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/lib
cp -r $TENSORFLOW_DIR/bazel-bin/tensorflow/libtensorflow_framework.so /usr/local/lib
Refresh shared library cache:

ldconfig
Copy C++ example label_image under tensorflow source tree to standalone folder:

cd /home/standalone
cp $TENSORFLOW_DIR/tensorflow/examples/label_image/main.cc .
cp -r $TENSORFLOW_DIR/tensorflow/examples/label_image/data .
Get inception model

cd /home/standalone/data
wget https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
tar -xzvf inception_v3_2016_08_28_frozen.pb.tar.gz
Create Makefile in standalone folder with content:

CC = g++
CFLAGS = -std=c++11 -g -Wall -D_DEBUG -Wshadow -Wno-sign-compare -w
INC = -I/usr/local/include/eigen3
INC += -I./include/third_party
INC += -I./include
INC += -I./include/nsync/public/
LDFLAGS =  -lprotobuf -pthread -lpthread
LDFLAGS += -ltensorflow_cc -ltensorflow_framework

all: main

main:
        $(CC) $(CFLAGS) -o main main.cc $(INC) $(LDFLAGS)
run:
        ./main --image=./data/grace_hopper.jpg --graph=./data/inception_v3_2016_08_28_frozen.pb --labels=./data/imagenet_slim_labels.txt
clean:
        rm -f main
Now do:

make
make run
Normally, it wil output:

./main --image=./data/grace_hopper.jpg --graph=./data/inception_v3_2016_08_28_frozen.pb --labels=./data/imagenet_slim_labels.txt
2017-11-19 13:10:23.989200: I tensorflow/core/platform/cpu_feature_guard.cc:137]
2017-11-19 13:10:25.205406: I main.cc:250] military uniform (653): 0.834306
2017-11-19 13:10:25.205491: I main.cc:250] mortarboard (668): 0.0218692
2017-11-19 13:10:25.205520: I main.cc:250] academic gown (401): 0.0103579
2017-11-19 13:10:25.205544: I main.cc:250] pickelhaube (716): 0.00800814
2017-11-19 13:10:25.205563: I main.cc:250] bulletproof vest (466): 0.00535088
For those who use docker image, now you can quit the interactive shell of docker and commit it so that it keeps all the things that you just did:

# On the interactive shell of docker do
exit
# On the local shell do the following to get docker container's id
docker ps -l
# Then commit
docker commit <container-id> <image-name:tag>
