#!/usr/bin/env python3
'''
Written by Justus Languell, 2020-2021, jus@gtsbr.org
See info.txt
'''

# Imports
import motors

import cv2
import numpy as np
import time
import wx
import pyautogui
import os
import argparse

# Define Constants


# Main Function, taking arg inputs
def main(color,flip,vid):

    
    cap = cv2.VideoCapture(vid) # Load Video Stream

    I=0
    while True:  # Main Loop 
        I+=1
        #i = I // 4
        i = int(color)
        print(i,end='\r')
        ret, frame = cap.read() # Get Frame
        if flip:
            frame = cv2.flip(frame,1) # Process Flip if Asked

        height, width, channels = frame.shape # Define Shape

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        lower_red = np.array([0,0,0])
        upper_red = np.array([i,i,i])
        
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame,frame, mask= mask)

        #cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        #cv2.imshow('res',res)

        #cv2.imshow('Happy New Year!', frame)  # Display Frame

        if cv2.waitKey(1) == ord('q'):  # Exit Condition
            break

    cv2.destroyAllWindows()
    pass

if __name__ == '__main__':
    # Setup Printout
    print(f'''
    Computer Vision Pipeline
    By Justus Languell 2020-2021
    ''')


    parser = argparse.ArgumentParser() # Process Args
    parser.add_argument('-f','--flip',help='Flip Video Stream',required=False,const=True,nargs='?')
    parser.add_argument('-c','--color',help='Color To Follow',required=False,default = '')
    parser.add_argument('-v', '--vid', help = 'Run Test on Video File', required = False, default = '')
    argument = parser.parse_args()

    flip = True if argument.flip else False
    vid = argument.vid if argument.vid else 0
    color = argument.color if argument.color else 'red'
        
    main(color,flip,vid) # Init Main Function
    print(color,flip,vid)
