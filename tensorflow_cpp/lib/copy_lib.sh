export TENSORFLOW_DIR=/workspace/software/tensorflow
sudo cp -r $TENSORFLOW_DIR/bazel-bin/tensorflow/libtensorflow.so ./
sudo cp -r $TENSORFLOW_DIR/bazel-bin/tensorflow/libtensorflow_cc.so ./
sudo cp -r $TENSORFLOW_DIR/bazel-bin/tensorflow/libtensorflow_framework.so ./
sudo cp -r $TENSORFLOW_DIR/bazel-bin/tensorflow/libtensorflow.so /usr/local/lib
sudo cp -r $TENSORFLOW_DIR/bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/lib
sudo cp -r $TENSORFLOW_DIR/bazel-bin/tensorflow/libtensorflow_framework.so /usr/local/lib


