from flask import Flask, request, Response
import json, os
from sound_module import SoundClassifier

app = Flask(__name__)
sounder = SoundClassifier()


@app.route('/api/sound', methods=['GET', 'POST'])
def sound():
    if request.method == 'GET':
        response = 'This is the SoundDescriber server created by Xufeng Zhao. Contact: xufeng.zhao@uni-hamburg.de'
    else:
        sounds = sounder.inference(**request.json)
        response = json.dumps(sounds)
    return Response(response=response, status=200, mimetype="application/json")


if __name__ =='__main__':
    app.run(host="0.0.0.0", port=8849)
