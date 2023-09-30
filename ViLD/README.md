# ViLD Server 

## Install 

### Install required packages
```bash
conda create -n vild python=3.9
conda activate vild
pip install -r requirements.txt
```

### Download weights
```bash
gsutil cp -r gs://cloud-tpu-checkpoints/detection/projects/vild/colab/image_path_v2 ./
```

# Launch ViLD Server
```bash
sh launch_vild_server.sh
```
The ViLD server will be ready under: `0.0.0.0:8848/api/vild`

## Call API with Python

```python
import requests
import json

addr = "YOUR_SERVER_ADDRESS:8848/api/vild"
# For example, on wtmgws10
# addr = "http://134.100.10.107:8848/api/vild"


# prepare headers for http request
content_type = "application/json"
headers = {"content-type": content_type}

data = {
    "image_path": "/informatik3/wtm/home/zhao/Codes/RLBench/examples/image.jpg",
    "category_names": [
        "red block",
        "blood colored block",
        "green block",
        "blue block",
        "orange block",
        "yellow block",
        "purple block",
    ],
    "detection_path":"./uids",
    "plot_on":True
}
data = json.dumps(data)
response = requests.post(addr, data=data, headers=headers)
# decode response
json.loads(response.text)

```
