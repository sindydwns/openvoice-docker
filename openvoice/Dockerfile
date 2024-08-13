FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt install git sudo curl gcc make unzip wget ffmpeg jupyter dumb-init -y
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
RUN /root/.pyenv/versions/3.9.19/bin/pip install ipykernel setuptools
RUN /root/.pyenv/versions/3.9.19/bin/python -m ipykernel install --user --name=project
RUN /root/.pyenv/versions/3.9.19/bin/pip install git+https://github.com/myshell-ai/MeloTTS.git
RUN /root/.pyenv/versions/3.9.19/bin/python -m unidic download
RUN wget https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
RUN git clone https://github.com/myshell-ai/OpenVoice.git
RUN /root/.pyenv/versions/3.9.19/bin/pip  install -r OpenVoice/requirements.txt

COPY install.sh .
RUN chmod +x install.sh && /bin/sh install.sh

WORKDIR /app/openvoice-repo

CMD ["/usr/bin/dumb-init", "--", "sleep", "infinity"]
