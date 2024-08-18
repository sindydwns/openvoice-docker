FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt install git vim sudo curl gcc make unzip wget ffmpeg jupyter dumb-init -y
RUN apt install build-essential libssl-dev zlib1g-dev libncurses5-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y
RUN curl https://pyenv.run | bash
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc; \
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc; \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
RUN /root/.pyenv/bin/pyenv install 3.9.19
RUN /root/.pyenv/bin/pyenv global 3.9.19

WORKDIR /app
RUN /root/.pyenv/versions/3.9.19/bin/pip install pandas gradio gradio_toggle

COPY app.py .
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["/usr/bin/dumb-init", "--", "sleep", "infinity"]
