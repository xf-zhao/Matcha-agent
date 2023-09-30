from flask import Flask, request, Response
import json, os
from vild import ViLDDetector


os.environ["DISPLAY"]=":1.0"

vilder = ViLDDetector()
# vilder = FakeViLDDetector()
app = Flask(__name__)

@app.route('/api/vild', methods=['GET', 'POST'])
def vild():
    if request.method == 'GET':
        response = 'This is the ViLD server created by Xufeng Zhao. Contact: xufeng.zhao@uni-hamburg.de'
    else:
        print(request.json)
        found_objects = vilder.detect(**request.json)
        response = json.dumps(found_objects)
    return Response(response=response, status=200, mimetype="application/json")


if __name__ =='__main__':
    app.run(host="0.0.0.0", port=8848)
