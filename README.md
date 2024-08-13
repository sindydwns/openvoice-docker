[openvoice](https://github.com/myshell-ai/OpenVoice)를 도커 위에서 사용하기 위한 프로젝트입니다.
---
다음 설치가 필요합니다.
- [Docker](https://www.docker.com)
- [Nvidia Driver](https://github.com/NVIDIA/nvidia-container-toolkit?tab=readme-ov-file)
    - troubleshooting
    만약 Failed to initialize NVML 에러가 발생하면 다음을 시도
    - vi /etc/nvidia-container-runtime/config.toml
    - no-cgroups 주석처리

make run
make exec
../install.sh