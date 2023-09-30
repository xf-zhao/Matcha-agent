# Install

```bash
conda create -n sound python=3.9
conda activate sound
pip install -r requirements.txt
```

# Offline Training

See also https://music-classification.github.io/tutorial/part3_supervised/tutorial.html

```bash
python train.py
```

What does this script do?
- Load the auditory `train/test` dataset (`.wav`)
- Train a neuralnetwork with augmented `train` dataset
- Evaluate on the `test` dataset
- Save the best performance model weights (`best_model.ckpt`), which will be loaded for the *sound server* as API.


# Launch Sound Server
```bash
sh launch_sound_server.sh
```
The sound server will be ready under: `0.0.0.0:8849/api/sound`