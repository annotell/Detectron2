FROM nvidia/cuda:13.3.0-cudnn-devel-ubuntu24.04@sha256:5c9fb04c50d925fc6a97739ee66f00f95e611fca1c82e6e84d9f560d61f3280e

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y python3.13 python3.13-dev build-essential ninja-build g++ curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13
RUN ln -sv /usr/bin/python3.13 /usr/bin/python

# Build dependencies - PyTorch must be present for CUDA extension compilation
ARG TORCH_VERSION=2.8.0
ARG CUDA_INDEX=cu128
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${CUDA_INDEX}"
RUN pip install --no-cache-dir setuptools wheel numpy

WORKDIR /build
COPY . .

ENV FORCE_CUDA=1
ARG TORCH_CUDA_ARCH_LIST="Pascal;Volta;Turing;Ampere;Ada;Hopper"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip wheel --no-build-isolation --no-deps -w /dist .
