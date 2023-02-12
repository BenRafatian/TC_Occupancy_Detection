import json
import cv2
import numpy as np
from PIL import Image as im

# resize the image (20 times scaling)
scale = 20  
width = scale * 32
height = scale * 24

# background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 16)

output_vid = cv2.VideoWriter('BG_sub.avi', 
                         cv2.VideoWriter_fourcc(*'DIVX'), 8.00, 
                         (width, height), False)

output_ma = cv2.VideoWriter('ma.avi', 
                         cv2.VideoWriter_fourcc(*'DIVX'), 8.00, 
                         (width, height), False)

ma_images_output = [] 
with open(".\imageData\ImageData500.json","r") as json_file:
    
    decodedArray = json.load(json_file)
    
    imageSetNumPyArray = np.asarray(decodedArray["data"])
    fgMask_images = []
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

        # cv2.imshow("test", grey_image)
        # cv2.waitKey(60)
        fgMask_images.append(fgMask)
        output_vid.write(fgMask)    

    for i in range(len(fgMask_images)-2):
        ma_images_output.append((fgMask_images[i] +
                                fgMask_images[i+1] +
                                fgMask_images[i+2] )
                                
                                // 3

        )
        output_ma.write(ma_images_output[i])
    print(len(ma_images_output)) 
    output_ma.release()   
    output_vid.release()