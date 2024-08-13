#!/bin/sh

mkdir /app/openvoice-repo -p
cp -r /app/OpenVoice/* /app/openvoice-repo
unzip /app/checkpoints_v2_0417.zip -d /app/openvoice-repo
pip install -e /app/openvoice-repo
echo "alias ju='jupyter notebook . --allow-root --ip 0.0.0.0 --no-browser'" >> ~/.bashrc
