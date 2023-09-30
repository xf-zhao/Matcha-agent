<div align="center">
<img src="https://matcha-agent.github.io/img/matcha_background.png" style="width:800px;"/>

Official Implementation of <b>Matcha Agent</b>

![](https://img.shields.io/badge/License-Apache-green) ![](https://img.shields.io/badge/Status-Full_Release-blue) ![](https://img.shields.io/badge/version-v1.0-blue)

---
</div>

### 🔔 News
- [2023-09-29] The **full** codes,including the NICOL robot URDF and scenes, are released! Codes are re-organised and tested with `Vicuna-13b` model.
- [2023-07-01] We open-source codes except the robot's configurations (because the NICOL robot is not publically available now). 


# Install
## Install RLBench and NICOL Robot
[NICOL](./NICOL/README.md)

## Install ViLD
[ViLD](./ViLD/README.md)

## Install Sound Module
[Sound Module](./Sound/README.md)


# Run
```bash
python main.py
```

Optional parameters:
- `engine`: The backend LLM to run, such as [`text-davinci-003`, `Vicuna-13b`, `gpt-3.5-turbo`, ...]


# Error Debuging
If error `ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found.`
see also: https://github.com/BVLC/caffe/issues/4953 

```bash
conda install libgcc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/${YOUR_USER_NAME}/anaconda3/envs/nicol/lib
```