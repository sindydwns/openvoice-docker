#!/bin/sh
git clone https://github.com/myshell-ai/OpenVoice
mv -r OpenVoice/* openvoice-repo/
pip install -r requirements.txt
pip install -e .
cd openvoice-repo
wget https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
unzip checkpoints_v2_0417.zip
rm checkpoints_v2_0417.zip

echo "alias ju='jupyter notebook . --allow-root --ip 0.0.0.0 --no-browser'" >> ~/.bashrc
