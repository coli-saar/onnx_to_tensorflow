# Convert ONNX models to Tensorflow

This is a simple Python 3 script which uses the [onnx-tensorflow backend](https://github.com/onnx/onnx-tensorflow) to convert an ONNX model into a Tensorflow model.

First, follow the installation instructions of onnx-tensorflow.

Then run onnx_to_tensorflow as follows:

```
python convert_onnx_tf.py model.onnx model.pb
```

You can choose whether the Tensorflow is saved in binary or text format by using the filename extension `.pb` or `.pbtxt`, respectively.



## Technical note

The Tensorflow file is written using the [write_graph](https://www.tensorflow.org/api_docs/python/tf/train/write_graph) method. This saves the graph structure and any constant weights, but not the weights in `Variable` nodes of the computation graph. As far as I can tell, onnx-tensorflow does not produce `Variable` nodes, so this seems fine; but I have not tested it thoroughly.



## Usage from Java

One usecase for this script is to be able to train a neural network with any framework that exports ONNX (e.g., PyTorch), and then run the trained model from Java, using the [Tensorflow Java binding](https://www.tensorflow.org/install/install_java). The file `OnnxLoader.java` in this repository illustrates how to do this.

