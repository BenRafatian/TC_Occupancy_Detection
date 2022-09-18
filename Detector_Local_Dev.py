# imports
import json
import time
from turtle import width
import cv2
import matplotlib.pyplot as plt
import numpy as np

# a variable to keep the count of current people inside the room.
CurrentPeopleInRoom = 0 

# resize the image (20 times scaling)
scale = 20  
width = scale * 32
height = scale * 24
# read the json image file and convert to NumPy array

with open("ImageData2.json","r") as json_file:
    
    decodedArray = json.load(json_file)
    
    imageSetNumPyArray = np.asarray(decodedArray["data"])
    
    for index, data in enumerate(imageSetNumPyArray):
        frame = data.get("Image")
        cc = str(frame)
        cc = cc[2:-6]
        data = np.fromstring(cc, sep=',')
        reshapedData = data.reshape(24, 32)

        min_value = np.min(data)
        max_value = np.max(data)
        # reshapedData = reshapedData - 
        # normalize image data between 0 - 255
        dim = (width, height)
        output = reshapedData * 255 / (max_value - min_value)
        new_imgGray = output.astype(np.uint8)
        img = cv2.applyColorMap(new_imgGray, cv2.COLORMAP_JET)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        new_imgGray = cv2.resize(new_imgGray, dim, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("output", img)
        cv2.waitKey(0)
        
        
# preview the raw image sequence

# continue...
