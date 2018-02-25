import argparse
import onnx
import os.path
import sys
from onnx_tf.backend import prepare
import tensorflow as tf


## Parse command-line arguments
argparser = argparse.ArgumentParser(description='')
argparser.add_argument("onnx_model", type=str, help="Name of ONNX model file (mandatory)")
argparser.add_argument("tf_model", type=str, help="Name of Tensorflow model file (mandatory)")

args = argparser.parse_args()


## Load ONNX model
model = onnx.load(args.onnx_model)
tf_rep = prepare(model)


## Print some information
# tf_rep.uninitialized = names of input nodes
# tf_rep.input_dict = dict from nodename to tensor
# tf_rep.predict_net.external_output = names of output nodes
print("Input placeholders:")
for input_name in tf_rep.uninitialized:
    it = tf_rep.input_dict[input_name]
    print("  %s, shape %s, %s" % (input_name, it.shape, it.dtype))

print("\nOutput tensors:")
for output_name in tf_rep.predict_net.external_output:
    ot = tf_rep.predict_net.output_dict[output_name]
    print("  %s, shape %s, %s" % (ot.name, ot.shape, ot.dtype))

sys.stdout.flush()

## Write graph
absolute_tf_path = os.path.realpath(args.tf_model)
dir, file = os.path.split(absolute_tf_path)
as_text = file.endswith(".pbtxt") # text = .pbtxt;   binary = .pb + as_text=False
print("\nWriting Tensorflow model as %s proto ..." % ("text" if as_text else "binary"))
sys.stdout.flush()

tf.train.write_graph(tf.get_default_graph(), dir, file, as_text=as_text)
print("Wrote Tensorflow model to %s." % absolute_tf_path)

