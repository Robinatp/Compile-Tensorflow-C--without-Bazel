export TENSORFLOW_DIR=/workspace/software/tensorflow
sudo cp -r $TENSORFLOW_DIR/bazel-bin/tensorflow/libtensorflow.so ./
sudo cp -r $TENSORFLOW_DIR/bazel-bin/tensorflow/libtensorflow_cc.so ./
#sudo cp -r $TENSORFLOW_DIR/bazel-bin/tensorflow/libtensorflow_framework.so ./
#bazel build //tensorflow:libtensorflow_cc.so
#bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.1 --copt=-msse4.2 --config=monolithic //tensorflow:libtensorflow_cc.so
sudo cp -r $TENSORFLOW_DIR/bazel-bin/tensorflow/libtensorflow.so /usr/local/lib
sudo cp -r $TENSORFLOW_DIR/bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/lib
#sudo cp -r $TENSORFLOW_DIR/bazel-bin/tensorflow/libtensorflow_framework.so /usr/local/lib


