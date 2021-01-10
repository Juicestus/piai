#!/usr/bin/env python3
'''
Written by Justus Languell, 2020-2021, jus@gtsbr.org
See info.txt
'''

# Imports
import cv2
import numpy as np
import time
import pyautogui
import os
import argparse
import time
import requests
import json
import io
import sys


#URL = 'http://fd63c2eba574.ngrok.io/detect'
URL = 'http://localhost:5000/detect'

SCALE_DOWN = .5

def print_motors(motors):
    out = ''
    for motor in motors:
        out += ('' + ('Forward ' if motor > 0 else 'Reverse ') if motor != 0 else 'Stopped ')
    print(out, end = '\r')

# Main Function, taking arg inputs
def main(flip,vid):


    fcount = 1
    cap = cv2.VideoCapture(vid) # Load Video Stream
    start = time.time()
    while True:  # Main Loop 
        ret, frame = cap.read() # Get Frame
        height, width, channels = frame.shape # Define Shape

        if SCALE_DOWN != 1:
            frame = cv2.resize(frame,(int(width*SCALE_DOWN),int(height*SCALE_DOWN)),interpolation=cv2.INTER_AREA)
            height, width, channels = frame.shape # Define Shape


        if flip:
            frame = cv2.flip(frame,1) # Process Flip if Asked

        frame = frame.astype(np.uint8) 

        framejson = json.dumps(frame.tolist())
        response = requests.post(URL,json=framejson)
        response = json.loads(response.text)

        if response['frame'] != 0:
            frame = np.array(json.loads(response['frame']))

        motors = [response['left'],response['right']]
        
        print_motors(motors)

        cv2.imshow('Happy New Year!', frame)  # Display Frame
        #time.sleep(5)
        fcount += 1

        if cv2.waitKey(1) == ord('q'):  # Exit Condition
            break

    end = time.time()
    elapsed = end-start
    print(f'Time Elapsed {elapsed}')
    print(f'AVG FPS: {fcount/elapsed}')

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Setup Printout
    parser = argparse.ArgumentParser() # Process Args
    parser.add_argument('-f', '--flip', help = 'Flip Video Stream', required = False, default = '')
    parser.add_argument('-v', '--vid', help = 'Run Test on Video File', required = False, default = '')
    argument = parser.parse_args()

    vid = 0
    flip = False
    if argument.flip:   # Process Args More
        flip = True
    if argument.vid:
        vid = argument.vid
        
    main(flip,vid) # Init Main Function
