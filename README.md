# Implementation of Matcha model


This is the implementation of paper [Chat with the Environment: Interactive Multimodal Perception using Large Language Models](https://arxiv.org/abs/2303.08268), or Matcha (multimodal environment chatting agent) for short.


This implementation is based on [RLBench](https://github.com/stepjam/RLBench) (in [CoppeliaSim](https://www.coppeliarobotics.com/) simulator) but with NICOL robot from our own lab. Because the full robot is not publically available, so in this code we open-source all the codes except the robot's configurations.


## Core elements are

- [ViLD](https://arxiv.org/abs/2104.13921) detection model to detect objects for describing and manipulating
- [OpenAI text-davinci-003 GPT model](https://openai.com/product) to generate planning decisions. Note that now there are options, e.g. GPT3.5 and GPT4, cheaper and better available now, which are not available on the paper sumission date.
- Multimodal classification models trained with supervised learning


## Install

ViLD reuqires many packages that might conflict with CoppeliaSim etc. So in this implementation, the vision and audio perception module are implemented as a separated http service with [Flask](https://flask.palletsprojects.com/en/2.3.x/).


The codes start from `main.py`.


Please cite
```text
@article{zhao2023chat,
  title={Chat with the Environment: Interactive Multimodal Perception using Large Language Models},
  author={Zhao, Xufeng and Li, Mengdi and Weber, Cornelius and Hafez, Muhammad Burhan and Wermter, Stefan},
  journal={arXiv preprint arXiv:2303.08268},
  year={2023}
}
```