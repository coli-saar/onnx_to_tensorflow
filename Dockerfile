FROM phusion/baseimage

RUN apt-get update
RUN apt-get install -y  git wget bzip2 python python3 python3-pip unzip

RUN pip3 install protobuf numpy tensorflow pybind11 pyyaml mkl-devel setuptools cmake cffi typing

# compile Pytorch from source
RUN git clone --recursive https://github.com/pytorch/pytorch
WORKDIR /pytorch
RUN python3 setup.py install

# get current protoc
WORKDIR /protoc
RUN wget https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-all-3.5.1.zip
RUN unzip protobuf-all-3.5.1.zip 
WORKDIR /protoc/protobuf-3.5.1
RUN ./configure
RUN make install
ENV LD_LIBRARY_PATH=/usr/local/lib

# compile ONNX from source
WORKDIR /
RUN git clone --recursive https://github.com/onnx/onnx.git
WORKDIR /onnx
RUN python3 setup.py install

WORKDIR /

# docker run -v ~/Documents/workspace/onnx_to_tensorflow/:/onnx_to_tensorflow -v ~/Documents/workspace/onnx-tensorflow:/onnx-tensorflow -it onnx bash 
