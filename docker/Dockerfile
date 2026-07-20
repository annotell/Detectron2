FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04@sha256:217134b60289ced62b47463be32584329baf35cb55a9ccda6626b91bd59803be

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && apt-get install -y python3-opencv ca-certificates python3.10 python3-pip git wget sudo ninja-build
RUN ln -sv /usr/bin/python3.10 /usr/bin/python

RUN pip install tensorboard cmake onnx
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
RUN pip install setuptools==59.5.0 opencv-python==4.5.5.64
RUN apt-get install libgl1 -y

RUN pip install 'git+https://github.com/facebookresearch/fvcore'

WORKDIR /root

COPY detectron2 /root/detectron2

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install -e detectron2

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

ENV PYTHONPATH "${PYTHONPATH}:/root/detectron2"

WORKDIR /root/detectron2

