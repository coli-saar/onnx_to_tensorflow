FROM phusion/baseimage

RUN apt-get update
RUN apt-get install -y  git wget


RUN wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
RUN bash Anaconda3-5.1.0-Linux-x86_64.sh

