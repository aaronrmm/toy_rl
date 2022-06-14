FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
#FROM deepspeed/rocm501:ds060_pytorch110

RUN pip install jupyterlab
RUN pip install transformers
RUN pip install datasets
RUN pip install deepspeed
RUN pip install "ray[rllib]" torch
RUN pip install "gym[atari]" "gym[accept-rom-license]" atari_py