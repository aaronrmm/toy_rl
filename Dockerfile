FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

RUN pip install jupyterlab
RUN pip install transformers
RUN pip install datasets
RUN pip install deepspeed
RUN pip install "ray[rllib]" torch
RUN pip install "gym[atari]" "gym[accept-rom-license]" atari_py