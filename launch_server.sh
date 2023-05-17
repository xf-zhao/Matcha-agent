LD_LIBRARY_PATH=/data/zhao/anaconda3/envs/saycan/lib/:$LD_LIBRARY_PATH \
      # DISPLAY=:1.7 \
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 \
        /data/zhao/anaconda3/envs/saycan/bin/python server.py