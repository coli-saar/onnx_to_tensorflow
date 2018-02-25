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









#
#
# sys.exit(0)
#
#
#
#
# print("graph: %s" % str(tf.get_default_graph()))
# print("ops: %s" % str(tf.get_default_graph().get_operations()))
#
# for op in tf.get_default_graph().get_operations():
#     print("---")
#     print(op)
#     print("input tensors: %s" % str(op.inputs))
#     print("output tensors: %s" % str(op.outputs))




# saver = tf.train.Saver()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# saver.save(sess, 'my_test_model')

# print(tf_rep.predict_net)
# print('-----')
# print(tf_rep.input_dict)
# print('-----')
# print("uninit %s" % tf_rep.uninitialized)
# print("-> %s" % tf_rep.input_dict[tf_rep.uninitialized[0]])
# print("out %s" % str(tf_rep.predict_net.external_output))
# print("outt %s" % str(tf_rep.predict_net.output_dict["12"]))


# print("inputs: %s" % tf_rep.predict_net.external_input)
# with tf.Session() as sess:
#     print("3: %s" % str(sess.run(tf_rep.input_dict["3"])))
# sys.exit(0)


#
# def make_null_input(input_dict):
#     for k,v in input_dict.items():
#         print("...")
#         print(k)
#         print(v)
#         print(type(v))

# make_null_input(tf_rep.input_dict)

# sys.exit(0)


# Add an op to initialize the variables.
# init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#
#     print("x")
#     inputs = tensors_to_ndarrays(sess, tf_rep.input_dict)
#     print("y")
#     print(inputs)
#
#     out = run(tf_rep, inputs)
#     print("z")
#     print(out)
#
#     # saver = tf.train.Saver()
#
#
#     sys.exit(0)
#
#
#
#
#
#
#
#
#     save_path = saver.save(sess, "/tmp/model.ckpt")
#     print("Model saved in path: %s" % save_path)
#
# # inp = make_null_input(tf_rep.input_dict)
#
# def tensors_to_ndarrays(sess, tensordict):
#     ret = {}
#     for k,v in tensordict.items():
#         # print("...")
#         # print(k)
#         # print(v)
#         shape = sess.run(tf.shape(v))
#         dtype = v.dtype.as_numpy_dtype
#         # print(str(dtype))
#         arr = np.zeros(shape, dtype=dtype)
#         # print(arr)
#         ret[k] = arr
#
#     return ret

#     sess.run(tf.global_variables_initializer())
#     external_output = dict(filter(lambda kv: kv[0] in tf_rep.predict_net.external_output, list(tf_rep.predict_net.output_dict.items())))
#     feed_dict = {tf_rep.input_dict[key]: feed_dict[key] for key in self.uninitialized}
#     output_values = sess.run(list(external_output.values()), feed_dict=feed_dict)
#
# print(tf_rep.predict_net.output_dict)