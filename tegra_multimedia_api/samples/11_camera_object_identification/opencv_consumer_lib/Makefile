###############################################################################
#
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

# TODO support cross-compilation

CC=g++
#CFLAGS+=-g

CFLAGS+= -fPIC -Wall -Wextra -O2 -g

CFLAGS+=`pkg-config --cflags opencv`
LDFLAGS+=`pkg-config --libs opencv`

CUDA_DIR:=/usr/local/cuda
CAFFE_DIR:=/home/ubuntu/Work/caffe/caffe-master
CFLAGS+=-I$(CAFFE_DIR)/include -I$(CUDA_DIR)/include -I$(CAFFE_DIR)/build/src
LDFLAGS+=-L$(CAFFE_DIR)/build/lib

TARGET_LIB=libopencv_consumer.so

LIB_SRCS=opencv_consumer.cpp Classifier.cpp
LIB_OBJS=$(LIB_SRCS:.cpp=.o)

.PHONY: all clean

all: $(TARGET_LIB)

$(TARGET_LIB): $(LIB_OBJS)
	$(CC) -o $(TARGET_LIB) $(LIB_OBJS) $(LDFLAGS) -shared -lcaffe

%.o: %.cpp
	$(CC) -c $(CFLAGS) $<

clean:
	rm -f $(LIB_OBJS) $(TARGET_LIB)
