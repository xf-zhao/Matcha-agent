conda install -c conda-forge qt=5.12.5
conda install opencv
conda install pillow
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm gym playsound imageio==2.4.1 imageio-ffmpeg==0.4.5 openai
pip install git+https://github.com/openai/CLIP.git
# 
pip install -U --no-cache-dir gdown --pre
pip install moviepy
pip install flax==0.5.3
pip install openai
pip install easydict
pip install imageio-ffmpeg
pip install tensorflow==2.7.0  # If error: UNIMPLEMENTED: DNN library is not found.

# ViLD pretrained model weights.
# pip install gsutil
# gsutil cp -r gs://cloud-tpu-checkpoints/detection/projects/vild/colab/image_path_v2 ./

# cd pyrep && pip install -r requirements && pip install -e .
# cd rlbench && pip install -r requirements && pip install -e .

# pip install protobuf==3.20 # protobuf<3.21

# if import cv2 lib error 
# export LD_LIBRARY_PATH=/export/home/zhao/anaconda3/envs/sound/lib:$LD_LIBRARY_PATH
