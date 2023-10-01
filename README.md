<div align="center">
<img src="https://matcha-agent.github.io/img/matcha_background_small.png" style="width:800px;"/>

Official Implementation of <a href="https://matcha-agent.github.io/"> <b>Matcha Agent</b> </a> 🍵~🤖 
![](https://img.shields.io/badge/License-Apache_2.0-green)
![](https://img.shields.io/badge/Status-Full_Release-blue)
![https://github.com/xf-zhao/Matcha-agent/releases/tag/v1.0](https://img.shields.io/badge/version-v1.0-blue)
![](https://img.shields.io/badge/Paper-Arxiv-blue)
![](https://img.shields.io/badge/Conference-IROS'23-forestgreen)

---
</div>

## 🔔 News
- $\color{red}{\text{[2023-09-29]}}$ The **full** codes,including the [NICOL](https://arxiv.org/abs/2305.08528) robot URDF and scenes, are released! Codes are re-organised and tested with `Vicuna-13b` model.
- $\text{[2023-07-01]}$ We open-source codes except the robot's configurations (because the [NICOL](https://arxiv.org/abs/2305.08528) robot is not publically available at this time). 

### Contents
<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->
- [🎥 Demo Video](#-demo)
- [⚙️ Install Dependencies](#-install-dependencies)
   * [🕹 Robotic](#-robotic)
      + [Install RLBench and NICOL Robot](#install-rlbench-and-nicol-robot)
      + [Run NICOL demo with RLBench tasks](#run-nicol-demo-with-rlbench-tasks)
   * [🌄️ Vision](#-vision)
      + [Install ViLD requirements](#install-vild-requirements)
      + [Launch Flask server for ViLD](#launch-flask-server-for-vild)
   * [🔉 Sound](#-sound)
      + [Install sound module requirements](#install-sound-module-requirements)
      + [Offline Neural Network Training for Sound Classification. ](#offline-neural-network-training-for-sound-classification)
      + [Launch sound module as a server](#launch-sound-module-as-a-server)
   * [🦙 Large Language Models (LLMs) Configuration](#-large-language-models-llms-configuration)
- [🍵~🤖 Run Matcha-agent](#-run-matcha-agent)
- [🐞 Error Debuging](#-error-debuging)
- [🔗 Citation](#-citation)

<!-- TOC end -->

## 🎥 Demo Video

[![Matcha-agent demo](https://markdown-videos-api.jorgenkh.no/url?url=https%3A%2F%2Fyoutu.be%2FrMMeMTWmT0k)](https://youtu.be/rMMeMTWmT0k)

- Matcha agent manipulates objects with different sound, weights and haptics to determine their materials.
- [NICOL](https://arxiv.org/abs/2305.08528) robot from [Knowledge Technology Group, University of Hamburg](https://www.inf.uni-hamburg.de/en/inst/ab/wtm/about.html).
- In [CoppeliaSim](https://www.coppeliarobotics.com/) simulator.
- Please **turn on** your speaker to hear the sound!

## ⚙️ Install Dependencies

### 🕹 Robotic

The experimental task is designed on top of [RLBench](https://github.com/stepjam/RLBench), but with a replacement of our own [NICOL](https://arxiv.org/abs/2305.08528) robot, a desktop-based humanoid robot. 

#### Install [RLBench](https://github.com/stepjam/RLBench) and [NICOL](./NICOL/README.md) Robot

```bash
git clone git@github.com:xf-zhao/Matcha-agent.git
cd NICOL
pip install -r requiremetns.txt
```

#### Run NICOL demo with [RLBench](https://github.com/stepjam/RLBench) tasks
```bash
python demo.py
```

### 🌄️ Vision

The visual detection is done with [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild), an open-vocabulary detection model. Despite of the simplicity of the vision in our demo, we use [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild) with a consideration of better generalization.

#### Install [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild) requirements

Since the library dependencies of [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild) may highly conflict with other packages installed, we encourage to install [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild) model within an separated environment and launch it as a `http` server.

```bash
conda create -n vild python=3.9
conda activate vild
pip install -r requirements.txt
# Download weights
gsutil cp -r gs://cloud-tpu-checkpoints/detection/projects/vild/colab/image_path_v2 ./
```

#### Launch [Flask](https://flask.palletsprojects.com/en/2.3.x/) server for [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)
```bash
sh launch_vild_server.sh
```
The [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild) server will be ready under: `0.0.0.0:8848/api/vild`

### 🔉 Sound

The sound module requires [PyTorch](https://pytorch.org/), [TorchAudio](https://pytorch.org/audio/stable/index.html) and other sound related packages that may conflict with the robotic and vision configurations. Like for vision module, we also deploy this module within an independent environment.

#### Install sound module requirements
```bash
conda create -n sound python=3.9
conda activate sound
pip install -r requirements.txt
```
#### Offline Neural Network Training for Sound Classification. 
We train a sound classification neural network.

```bash
python train.py
```
This training process includes
- Load the auditory `train/test` dataset (`.wav`)
- Train a neuralnetwork with augmented `train` dataset
- Evaluate on the `test` dataset
- Save the best performance model weights (`best_model.ckpt`), which will be loaded for the *sound server* as API.
See also [this blog](https://music-classification.github.io/tutorial/part3_supervised/tutorial.html) for reference.

#### Launch sound module as a server
```bash
sh launch_sound_server.sh
```
The sound server will be ready under: `0.0.0.0:8849/api/sound`

### 🦙 Large Language Models (LLMs) Configuration

In the original Matcha-agent paper, we use openai API `text-davinci-003` and `text-ada-001` as the backend LLMs. Nowadays, there are many open-sourced LLMs available. In the version `v1.0` release, we use `Vicuna-13b` model followed with [this FastChat doc](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md).

*Note that the LLM is worked in a **completions** mode instead of **chat completions** mode, i.e. no role-plays since we manually introduce roles in the prompts.*

## 🍵~🤖 Run Matcha-agent
```bash
python main.py
```

Optional parameters:
- `engine`: The backend LLM to run, such as [`text-davinci-003`, `Vicuna-13b`, `gpt-3.5-turbo`, ...]
...


## 🐞 Error Debuging
- If an error `ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found.` occurs:

    ```bash
    conda install libgcc
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/${YOUR_USER_NAME}/anaconda3/envs/nicol/lib
    ```
    see also: https://github.com/BVLC/caffe/issues/4953 


## 🔗 Citation
```text
@misc{zhao2023chat,
      title={Chat with the Environment: Interactive Multimodal Perception Using Large Language Models}, 
      author={Xufeng Zhao and Mengdi Li and Cornelius Weber and Muhammad Burhan Hafez and Stefan Wermter},
      year={2023},
      eprint={2303.08268},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

<!-- ## 👥 Contributors
<a href="https://github.com/xf-zhao/Matcha-agent/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xf-zhao/Matcha-agent" />
</a> -->