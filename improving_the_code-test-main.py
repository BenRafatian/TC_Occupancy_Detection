from __future__ import print_function

import json
import time

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
#import IoT hub library

scalling = 20
width = scalling * 32
height = scalling * 24
backSub = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold = 16)

# add device management to initialize the code
img = np.zeros([height, width, 3])

imgGray = np.zeros([height, width, 3])
                

time.sleep(1)
cx = 0
r1 = 0
r2 = 0
r3 = 0
frame = [0] * 768
current_region = np.array([0, 0, 0])
per_region = np.array([0, 0, 0])
totalNofPeople = 0


frame = [0] * 768
with open("ImageData2.json", "r") as read_file:
    print("Converting JSON encoded data into Numpy array")
    decodedArray = json.load(read_file)

    finalNumpyArrayOne = np.asarray(decodedArray["data"])
    print("NumPy Array One")
    #  print(finalNumpyArrayOne)
    for index, data in enumerate(finalNumpyArrayOne):
        if index ==0:
            frame = data.get("Image")
            # read data from serial
            # print(np.sum(frame),index)
            cc = str(frame)
            cc = cc[2:-6]
            data = np.fromstring(cc, sep=',')
            # reshape data into matrix
            output_p = data.reshape(24, 32)
        else :
            frame = data.get("Image")
            # read data from serial
            # print(np.sum(frame),index)
            cc = str(frame)
            cc = cc[2:-6]
            data = np.fromstring(cc, sep=',')
            # reshape data into matrix
            output_c = data.reshape(24, 32)
            output = (output_c + output_p) / 2
            output_p = output_c
            # scaling
            minValue = 50 # math.floor(np.amin(output))
            maxValue = 100
            output = output - minValue
            output = output * 255 / (maxValue - minValue)  # Now scaled to 0 - 255


            # resize image
            dim = (width, height)
            # output = cv2.resize(output, dim, interpolation = cv2.INTER_LINEAR )
            # apply colormap
            new_imgGray = output.astype(np.uint8)
            img = cv2.applyColorMap(new_imgGray, cv2.COLORMAP_JET)




            img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

            new_imgGray = cv2.resize(new_imgGray, dim, interpolation=cv2.INTER_LINEAR)

            fgMask = backSub.apply(new_imgGray) #background Subtraction applied
            if index < 20:
                continue
            #kernel
            kernel = np.ones((6,6), np.uint8)
            fgMask = cv2.dilate(fgMask,kernel, iterations=4)
            fgMask = cv2.erode(fgMask, kernel, iterations=4)


            fgMask = fgMask + cv2.Canny(fgMask, 50, 70, L2gradient=True)

            ###############   finding contours ################
            contours, _ = cv2.findContours(fgMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for inx, cnt in enumerate(contours):

                # Calculate area and remove small elements
                area = cv2.contourArea(cnt)

                if area > 200*scalling:
                    # cv2.drawContours(img, [cnt], -1, (0,255,0), 10)
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx = int((x + x + w) / 2)
                    cy = int((y + y + h) / 2)
                    # print(area)
                    cv2.circle(img, (cx,cy), 10 , (0,255,0), -1)
                    # v2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)
                    cv2.drawContours(img, contours, inx, (0, 0, 255), 3)
                    img = cv2.line(img, (int(2*img.shape[1]/3),0), (int(2*img.shape[1]/3),img.shape[0]), (255,255,255), 2)
                    img = cv2.line(img, (int(img.shape[1]/3), 0), (int(img.shape[1]/3), img.shape[0]), (255, 255, 255), 2)

                    print("X:", cx, "Y:", cy)
                    print(img.shape)
                # updating number of people in each region of interest
                if 0 <= cx <= img.shape[1]/3:
                    r1 = r1 + 1
                if img.shape[1]/3 <= cx <= 2 * img.shape[1]/3:
                    r2 = r2 + 1
                if  2 * img.shape[1]/3 <= cx:
                    r3 = r3 + 1
                current_region = np.array([r1, r2, r3])
                r1 = 0
                r2 = 0
                r3 = 0
                print("per_region: ", per_region, "current region: ", current_region)

            if current_region[1] != per_region[1]:
                if current_region[0] > per_region[0]:
                    totalNofPeople = totalNofPeople + 1
                if current_region[2] > per_region[2]:
                    totalNofPeople = totalNofPeople -1

                # send data to the azure IoT hub

            if current_region is not np.array([0, 0, 0]):
                per_region = current_region
                print("all not-zero current to previous")
            print("current people in the room: ", totalNofPeople)

            ########################
            text = "Min: " + str(minValue) + " C  Max: " + str(maxValue) + " C"
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (20, 50)
            image = cv2.putText(img, text, org, font, 1, (255, 255, 255), 2, cv2.LINE_AA)


            cv2.imshow("image", img)
            #cv2.imshow('FG Mask1', fgMask1)
            cv2.imshow("gray", new_imgGray)
            # cv2.imshow('FG Mask2', fgMask2)
            cv2.imshow("Bsubtraction",fgMask)
            cv2.waitKey(0)
