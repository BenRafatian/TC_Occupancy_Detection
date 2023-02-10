# imports
import json
import time
from turtle import width
import cv2
import matplotlib.pyplot as plt
import numpy as np
from manual_tracker import *


# a variable to keep the count of current people inside the room.
CurrentPeopleInRoom = 0 

# variables for each zone of the picture to keep track of the number of their objects at the moment
zone1 = 0
zone2 = 0
zone3 = 0

# 2 arrays for keeping the track of zone counts for the current frame and th eprevious frame
current_frame = np.array([0, 0, 0])
previous_frame = np.array([0, 0, 0])


# resize the image (20 times scaling)
scale = 20  
width = scale * 32
height = scale * 24

# background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 16)

# read the json image file and convert to NumPy array

with open(".\imageData\ImageData500.json","r") as json_file:
    
    decodedArray = json.load(json_file)
    
    imageSetNumPyArray = np.asarray(decodedArray["data"])
    
    for index, data in enumerate(imageSetNumPyArray):
        frame = data.get("Image")
        cc = str(frame)
        cc = cc[2:-6]
        data = np.fromstring(cc, sep=',')
        reshapedData = data.reshape(24, 32)

        # max_value = np.max(data)
        max_value = 100
        # min_value = np.min(data)        
        min_value = 50   
        # reshapedData = reshapedData - 
        # normalize image data between 0 - 255
        dim = (width, height)
        output = reshapedData * 255 / (max_value - min_value)
        
        # create the grey image and resize it for preview
        grey_image = output.astype(np.uint8)
        grey_image = cv2.resize(grey_image, dim, interpolation=cv2.INTER_LINEAR)


        # color map the image to get a heatmap
        colored_img = cv2.applyColorMap(grey_image, cv2.COLORMAP_JET)
        
        # Background Subtraction Process to create a ForeGround mask
        # !!! Take a look at this Part later for improvement !!!

        fgMask = backSub.apply(grey_image)
        
        #kernel
        kernel = np.ones((6,6), np.uint8)
        fgMask = cv2.dilate(fgMask,kernel, iterations=4)
        fgMask = cv2.erode(fgMask, kernel, iterations=4)
        fgMask = fgMask + cv2.Canny(fgMask, 50, 70, L2gradient=True)

        # Finding and Drawing contours
        contours, _ = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for index, contour in enumerate(contours):

            #calculate the area and removing the small elements
            area = cv2.contourArea(contour)
            
            if area > 200 * scale:
                x, y, w, h = cv2.boundingRect(contour)
                cx = int((x + x + w) / 2)
                cy = int((y + y + h) / 2)

                # Draw bounding boxes and their centers on all three versions of image
                cv2.circle(grey_image, (cx,cy), 10 , (0,255,0), -1)
                cv2.rectangle(grey_image, (x, y), (x + w, y + h), (255, 255, 0), 3)
                cv2.drawContours(grey_image, contours, index, (0, 0, 255), 3)

                cv2.circle(colored_img, (cx,cy), 10 , (0,255,0), -1)
                cv2.rectangle(colored_img, (x, y), (x + w, y + h), (255, 255, 0), 3)
                cv2.drawContours(colored_img, contours, index, (0, 0, 255), 3)

                cv2.circle(fgMask, (cx,cy), 10 , (0,255,0), -1)
                cv2.rectangle(fgMask, (x, y), (x + w, y + h), (255, 255, 0), 3)
                cv2.drawContours(fgMask, contours, index, (0, 0, 255), 3)


                ####### Counting Procedure #######

                # Set zones and update the count of current objects in each zone
                if 0 <= cx < colored_img.shape[1]/3:
                    zone1 += 1

                if colored_img.shape[1]/3 <= cx < 2 * colored_img.shape[1]/3:
                    zone2 += 1

                if 2 * colored_img.shape[1]/3 <= cx:
                    zone3 += 1

                # update current frame count array and reset the zone counts
                current_frame = ([zone1, zone2, zone3])
                zone1 = 0
                zone2 = 0
                zone3 = 0     

                # print out the current frame objects count and the previous frame count arrays
                print("Previous Frame count array:", previous_frame, 
                        "Current Frame count array:", current_frame)
                
                # check which zone has been updated between two frames and with that update total number 

                """
                Huge Problem with this method 
                doesn't work when a series of people movement syncs with frames of the picture
                """
                if current_frame[1] != previous_frame[1]:
                    if current_frame[0] > previous_frame[0]:
                        CurrentPeopleInRoom -= 1

                    if current_frame[2] > previous_frame[2]:
                        CurrentPeopleInRoom += 1    

                # now we should set the current frame to previous
                previous_frame = current_frame

                # we print out the current people in the room
                print("Current people in the room:", CurrentPeopleInRoom)        



        #preview Grey, Colored, and BG subtracted image
        # cv2.imshow("Grey Image", grey_image)
        cv2.imshow("Colored Image", colored_img)
        # cv2.imshow("BG Subtracted", fgMask)
        cv2.waitKey(0)


        
        

 
# continue...
