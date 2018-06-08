import argparse
import onnx
import os.path
import sys
from onnx_tf.backend import prepare
import numpy as np
import tensorflow as tf


# Test inputs
TEST_INPUTS = np.array([[[-0.5525, 0.6355, -0.3968]],[[-0.6571, -1.6428, 0.9803]],[[-0.0421, -0.8206, 0.3133]],[[-1.1352, 0.3773, -0.2824]],[[-2.5667, -1.4303, 0.5009]]])
TEST_INITIAL_H = np.array([[[0.5438, -0.4057,  1.1341]]])
TEST_INITIAL_C = np.array([[[-1.1115, 0.3501, -0.7703]]])


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


## Parse command-line arguments
argparser = argparse.ArgumentParser(description='')
argparser.add_argument("onnx_model", type=str, help="Name of ONNX model file (mandatory)")
argparser.add_argument("tf_model", type=str, help="Name of Tensorflow model file (mandatory)")

args = argparser.parse_args()


## Load ONNX model
model = onnx.load(args.onnx_model)
tf_rep = prepare(model)


'''
## Print some information
# tf_rep.uninitialized = names of input nodes
# tf_rep.input_dict = dict from nodename to tensor
# tf_rep.predict_net.external_output = names of output nodes
print("Input placeholders:")
for input_name in tf_rep.predict_net.external_input:
    it = tf_rep.input_dict[input_name]
    print("  %s, shape %s, %s" % (input_name, it.shape, it.dtype))

print("\nInput dictionary:")
for input_name in tf_rep.input_dict:
    it = tf_rep.input_dict[input_name]
    print("  %s, shape %s, %s" % (input_name, it.shape, it.dtype))

print("\nOutput tensors:")
for output_name in tf_rep.predict_net.external_output:
    ot = tf_rep.predict_net.output_dict[output_name]
    print("  %s, shape %s, %s" % (ot.name, ot.shape, ot.dtype))

sys.stdout.flush()
'''

## Write graph
absolute_tf_path = os.path.realpath(args.tf_model)
dir, file = os.path.split(absolute_tf_path)
as_text = file.endswith(".pbtxt") # text = .pbtxt;   binary = .pb + as_text=False
print("\nWriting Tensorflow model as %s proto ..." % ("text" if as_text else "binary"))
sys.stdout.flush()

tf.train.write_graph(tf_rep.predict_net.graph.as_graph_def(), dir, file, as_text=as_text) # Export ohne Variablen?! Vorsicht!

print("Wrote Tensorflow model to %s." % absolute_tf_path)

graph = load_graph(absolute_tf_path)

print("Loaded Tensorflow model from %s." % absolute_tf_path)

with tf.Session(graph=graph) as sess:
    out = sess.run("prefix/Squeeze_3:0", feed_dict={"prefix/0:0": TEST_INPUTS, "prefix/1:0": TEST_INITIAL_H, "prefix/2:0": TEST_INITIAL_C})
    print(out)

