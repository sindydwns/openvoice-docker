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
COPY install.sh .

RUN /root/.pyenv/versions/3.9.19/bin/pip install ipykernel
RUN apt install jupyter -y
RUN /root/.pyenv/versions/3.9.19/bin/python -m ipykernel install --user --name=project

RUN apt install unzip wget ffmpeg -y
RUN /root/.pyenv/versions/3.9.19/bin/pip install git+https://github.com/myshell-ai/MeloTTS.git
RUN /root/.pyenv/versions/3.9.19/bin/python -m unidic download

RUN chmod +x install.sh && /bin/sh install.sh
RUN /root/.pyenv/versions/3.9.19/bin/pip install setuptools
RUN /root/.pyenv/versions/3.9.19/bin/pip install -r requirements.txt
RUN /root/.pyenv/versions/3.9.19/bin/pip install -e .

WORKDIR /app/openvoice-repo
RUN apt install dumb-init -y

CMD ["/usr/bin/dumb-init", "--", "sleep", "infinity"]
