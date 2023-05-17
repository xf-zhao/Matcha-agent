from flask import Flask, request, Response
import json, os
from vild import ViLDDetector, FakeViLDDetector
from sound_module import SoundClassifier

app = Flask(__name__)
# app.config["DISPLAY"]=":1.0"
# app.config['CUDA_VISIBLE_DEVICES'] = "7"
# app.config['CUDA_DEVICE_ORDER'] ="PCI_BUS_ID" 
# os.environ["DISPLAY"]=":1.0"
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="7"

# LD_LIBRARY_PATH=/data/zhao/anaconda3/envs/saycan/lib/:$LD_LIBRARY_PATH \
#       DISPLAY=:1.0 \
#         CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 \
#         python server.py

# use conda install tensorflow instead of pip!!!
vilder = ViLDDetector()
sounder = SoundClassifier()
# vilder = FakeViLDDetector()


@app.route('/api/vild', methods=['GET', 'POST'])
def vild():
    if request.method == 'GET':
        response = 'This is the ViLD server created by Xufeng Zhao. Contact: xufeng.zhao@uni-hamburg.de'
    else:
        found_objects = vilder.detect(**request.json)
        response = json.dumps(found_objects)
    return Response(response=response, status=200, mimetype="application/json")


@app.route('/api/sound', methods=['GET', 'POST'])
def sound():
    if request.method == 'GET':
        response = 'This is the SoundDescriber server created by Xufeng Zhao. Contact: xufeng.zhao@uni-hamburg.de'
    else:
        sounds = sounder.inference(**request.json)
        response = json.dumps(sounds)
    return Response(response=response, status=200, mimetype="application/json")


if __name__ =='__main__':
    app.run(host="0.0.0.0", port=8848)
