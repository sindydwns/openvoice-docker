FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt install git sudo curl -y
RUN apt install libbz2-dev libncurses5-dev libffi-dev libreadline-dev libsqlite3-dev liblzma-dev -y
RUN curl https://pyenv.run | bash
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc; \
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc; \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
RUN apt install gcc -y
RUN apt install make -y
RUN apt install build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y
RUN /root/.pyenv/bin/pyenv install 3.9.19
RUN /root/.pyenv/bin/pyenv global 3.9.19

WORKDIR /app
RUN git clone https://github.com/myshell-ai/OpenVoice

WORKDIR /app/OpenVoice
RUN /root/.pyenv/versions/3.9.19/bin/pip install setuptools
RUN /root/.pyenv/versions/3.9.19/bin/pip install -r requirements.txt
RUN /root/.pyenv/versions/3.9.19/bin/pip install -e .

RUN apt install unzip wget -y
RUN wget https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip \
    && unzip checkpoints_v2_0417.zip \
    && rm checkpoints_v2_0417.zip

RUN apt install dumb-init -y
RUN ["/usr/bin/dumb-init", "--", "sleep", "infinity"]
