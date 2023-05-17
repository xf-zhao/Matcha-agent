# python main.py -l -a
python main.py -l


# LD_LIBRARY_PATH=/data/zhao/anaconda3/envs/rlb/lib/:$LD_LIBRARY_PATH \
DISPLAY=:1.6 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
/data/zhao/anaconda3/envs/rlb/bin/python main.py -l -g text-davinci-003


DISPLAY=:1.6 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
/data/zhao/anaconda3/envs/rlb/bin/python main.py -a -l -g text-davinci-003