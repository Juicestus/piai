#!/usr/bin/env python3

import requests
import json
import numpy as np

URL = 'http://localhost:5000/detect'
#data = {'frame': arr.tolist()}
frame = np.random.rand(10,10)
framejson = json.dumps(frame.tolist())
reqret = requests.post(URL,json=framejson)
print(reqret)

