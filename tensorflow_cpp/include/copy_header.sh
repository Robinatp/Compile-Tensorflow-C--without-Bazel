export TENSORFLOW_DIR=/workspace/software/tensorflow
mkdir -p third_party
sudo cp -r $TENSORFLOW_DIR/tensorflow third_party/
sudo cp -r $TENSORFLOW_DIR/bazel-genfiles/tensorflow third_party/
sudo cp -r $TENSORFLOW_DIR/third_party/eigen3 third_party/

#export PROTOBUF_DIR=/workspace/software/protobuf
#cp -r PROTOBUF_DIR/src/google third_party/


#or I strongly suggest use the script as bellows:
export PROTOBUF_DIR=/workspace/software/tensorflow/tensorflow/contrib/makefile/
##cd PROTOBUF_DIR
##sudo ./build_all_linux.sh
##cd -
sudo cp -r $PROTOBUF_DIR/gen/protobuf/include/google third_party/


export EIGEN_DIR=/workspace/software/eigen-eigen-5a0156e40feb
sudo cp -r $EIGEN_DIR/. third_party/eigen3/
sudo cp -r third_party/eigen3/Eigen third_party/


git clone https://github.com/google/nsync.git
