#!/usr/bin/env python3
'''
Written by Justus Languell, 2020-2021, jus@gtsbr.org
See info.txt
'''

# Imports
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import sys
import string
import re
import pandas as pd
import os
from hashlib import sha256
import cv2
import numpy as np
import time
import pyautogui
import argparse
import json
import io

# Define Constants
TARGET_LIST = ['person']
MODEL_PATH = 'assets/frozenModel01.pb'
CONFIG_PATH = 'assets/frozenModel01.pbtxt'
LABELS_PATH = 'assets/labels.txt'
SCORE_THRESHOLD = 0.3
NETWORK_INPUT_SIZE = (300, 300)
NETWORK_SCALE_FACTOR = 1
MEAN_TUPLE = (127.5, 127.5, 127.5)
CBW = 30
CBH = 30
RETFRAME = False

app = Flask(__name__,template_folder='html') # Gen Flask App

with open(LABELS_PATH, 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')  # Get Labels

COLORS = np.random.uniform(0, 255, size=(len(labels), 3)) # Assing Random Colors
ssd_net = cv2.dnn.readNetFromTensorflow(model=MODEL_PATH, config=CONFIG_PATH) # Load Neural Network


def process(frame,modframe=False):
        height, width, channels = frame.shape # Define Shape
        frame = frame.astype('float32')
        blob = cv2.dnn.blobFromImage(image=frame,                               # Get Blob
                                     scalefactor=NETWORK_SCALE_FACTOR,
                                     size=NETWORK_INPUT_SIZE,
                                     mean=MEAN_TUPLE, crop=False)

        ssd_net.setInput(blob)   # Input Blob to NN
        network_output = ssd_net.forward()  # Get Output of NN
    
        cx = int(width/2)
        cy = int(height/2)
        cleft = int(cx-CBW)   # Center Box Math
        ctop = int(cy-CBH)
        cright = int(cx+CBW)
        cbottom = int(cy+CBH)

        if modframe:
            cv2.rectangle(frame, (cleft,ctop), (cright,cbottom), (0,0,255), 2) # Draw Centr Box

        motors = [0,0]  # Motor Actions Preset to LOW

        for detection in network_output[0,0]:   # Parse Through Output
            score = float(detection[2])
            class_index = np.int(detection[1])  # Finds Objects
            rawlabel = labels[class_index]
            label = f'{rawlabel}: {score:.2%}'
        
            if score > SCORE_THRESHOLD:     # Checks to be sure

                left = np.int(detection[3] * width)
                top = np.int(detection[4] * height) # Box Drawing Math
                right = np.int(detection[5] * width) 
                bottom = np.int(detection[6] * height)
                
                if rawlabel in TARGET_LIST:   # If a Target
     
                    label = f'Target: {rawlabel} {score:.2%}'
                    _left = int((right-left)/2) + left - 1      # Math to Check for Intersection
                    _right = int((right-left)/2) + left + 1

                    if modframe:
                        cv2.rectangle(frame,(_left,top),(_right,bottom), # Center Line
                                  COLORS[class_index],thickness=4,
                                  lineType=cv2.LINE_AA)
                        
                    is_right = not(_left < cright )    #  More Math to Check for Intersection
                    is_left = not(_right > cleft)

                    if not is_right and not is_left:       #  Processing Intrsection Data
                        motors = [1,1]

                    if is_right and not is_left:
                        if modframe:
                            cv2.arrowedLine(frame,(cleft,cy),(cright,cy),
                                        (0,0,255),6,tipLength=.5)
                        motors = [1,-1] 

                    if is_left and not is_right:
                        if modframe:
                            cv2.arrowedLine(frame,(cright,cy),(cleft,cy),
                                         (0,0,255),6,tipLength=.5)
                        motors = [-1,1]

                    if is_right and is_left:
                        pass

                if modframe:
                    cv2.rectangle(frame,(left,top),(right,bottom),   # Draws Box
                              COLORS[class_index],thickness=4,
                              lineType=cv2.LINE_AA)
                    cv2.putText(img=frame,   # Labels
                            text = label,
                            org=(left, np.int(top*0.9)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 2,
                            color=COLORS[class_index],
                            thickness = 2,
                            lineType=cv2.LINE_AA)

            if modframe:
                cv2.rectangle(frame,(0,height),(50,height-100),(255,0,0),4)  # Display Left Wheel

                if motors[0] == 1:
                    cv2.arrowedLine(frame,(25,height),(25,height-100),
                                    (0,0,255),6,tipLength=.25)
                if motors[0] == -1:
                    cv2.arrowedLine(frame,(25,height-100),(25,height),
                                    (0,0,255),6,tipLength=.25)
                else:
                    cv2.arrowedLine(frame,(25,height-100),(25,height),
                                    (0,0,255),6,tipLength=0)

                cv2.rectangle(frame,(width-50,height),(width,height-100),(255,0,0),4)  # Display Right Wheel

                if motors[1] == 1:
                    cv2.arrowedLine(frame,(width-25,height),(width-25,height-100),
                                    (0,0,255),6,tipLength=.25)
                if motors[1] == -1:
                    cv2.arrowedLine(frame,(width-25,height-100),(width-25,height),
                                    (0,0,255),6,tipLength=.25)
                else:
                    cv2.arrowedLine(frame,(width-25,height-100),(width-25,height),
                                    (0,0,255),6,tipLength=0)

                
        return motors,frame

@app.route('/')
def index():
    return '<h1>DNN REST Server</h1>'

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        frame = np.array(json.loads(request.json))
        motors, frame = process(frame, modframe=False)  # Pass frame through analysis
        response = {'left': motors[0], 'right': motors[1], 'frame': 0} 
        if RETFRAME:
            framejson = json.dumps(frame.tolist())
            response = {'left': motors[0], 'right': motors[1], 'frame': framejson} 

        return response


if __name__ == '__main__':
    app.run(debug=True)



