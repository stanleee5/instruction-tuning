FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y wget curl tree git vim zsh
RUN chsh -s $(which zsh)
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t linuxonly -p autopep8 -p git -p zsh-autosuggestions
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

WORKDIR /app
COPY ./configs /app
COPY ./src /app
COPY ./train.py /app
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
