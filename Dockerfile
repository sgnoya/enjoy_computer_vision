FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ARG USER_NAME
ARG UID
ARG GID
ARG GROUP_NAME

RUN apt update
RUN apt install -y sudo git

# base
RUN pip install prospector black matplotlib ipykernel

# stable diffusion
RUN pip install diffusers[torch]==0.9 transformers
RUN pip install --upgrade --pre triton
RUN pip install --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate scipy

# add sudo user
RUN groupadd -g ${GID} ${GROUP_NAME}
RUN useradd -ms /bin/sh -u ${UID} -g ${GID} ${USER_NAME}

RUN echo "${USER_NAME}:hogehoge" | chpasswd
RUN echo "${USER_NAME}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers

RUN apt install -y vim
# USER ${USER_NAME}
