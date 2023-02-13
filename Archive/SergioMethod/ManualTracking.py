import cv2
from tracker import *
import json
import numpy as np
# Create tracker object

tracker = EuclideanDistTracker()

# resize the image (20 times scaling)
scale = 20  
width = scale * 32
height = scale * 24

# background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 16)

# read the json image file and convert to NumPy array

with open(".\ImageData500.json","r") as json_file:
    
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

        # Background Subtraction Process to create a ForeGround mask
        # !!! Take a look at this Part later for improvement !!!

        fgMask = backSub.apply(grey_image)
        
        #kernel
        kernel = np.ones((6,6), np.uint8)
        fgMask = cv2.dilate(fgMask,kernel, iterations=4)
        fgMask = cv2.erode(fgMask, kernel, iterations=4)
        fgMask = fgMask + cv2.Canny(fgMask, 50, 70, L2gradient=True)
        
        roi = fgMask[90:390, 110:510]             
        # Finding and Drawing contours
        contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for index, contour in enumerate(contours):
            
            #calculate the area and removing the small elements
            area = cv2.contourArea(contour)
            
            if area > 200 * scale:
                x, y, w, h = cv2.boundingRect(contour)


                detections.append([x, y, w, h])

        # 2. Object Tracking
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 255, 0), 3)

        cv2.imshow("Mask", fgMask)

        cv2.waitKey(1)