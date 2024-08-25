FROM nvidia/cuda:12.6.0-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive 
ENV COPPELIASIM_ROOT=/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu20_04
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
COPY system/usrset.txt $COPPELIASIM_ROOT/system/usrset.txt
COPY ./ /Matcha-agent/
# COPY CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu20_04.tar.xz /
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y git zsh x11-apps python3 python3-pip wget \
&& apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libgl1-mesa-dev \
&& apt-get install -y libxcb-xinerama0 \
&& apt-get install -y libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 libegl1-mesa \
# && git clone --depth 1 https://github.com/xf-zhao/Matcha-agent.git \
&& git clone --depth 1 https://github.com/stepjam/RLBench.git \
&& echo "export COPPELIASIM_ROOT=/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu20_04" >> ~/.bashrc \
&& echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT" >> ~/.bashrc \
&& echo "export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT" >> ~/.bashrc \
&& source ~/.bashrc \
&& apt-get update && apt-get install ffmpeg libsm6 libxext6  -y \
# && wget --no-check-certificate https://downloads.coppeliarobotics.com/V4_5_1_rev4/CoppeliaSim_Edu_V4_5_1_rev4_Ubuntu20_04.tar.xz \
&& tar -xf /Matcha-agent/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu20_04.tar.xz \
&& pip3 install /RLBench \
&& pip3 install -r /Matcha-agent/NICOL/requirements.txt
# # Until now run `python3 /RLBench/examples/single_task_rl.py` to test if rlbench installed properly
WORKDIR /Matcha-agent
