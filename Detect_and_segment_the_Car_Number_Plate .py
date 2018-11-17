import numpy as np
import cv2
import  imutils
import sys
import os
import pandas as pd
from operator import itemgetter
from numpy import vstack
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\luan1\Anaconda3\Lib\site-packages'
image = cv2.imread(r'C:\Users\luan1\Desktop\Car_Image_1.jpg')

# Resize the image - change width to 500
image = imutils.resize(image, width=500)

# Display the original image
cv2.imshow("Original Image", image)

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Grayscale image", gray)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
#cv2.imshow("Bilateral image", gray)

# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)
#cv2.imshow("Canny edge image", edged)
image_copy = edged.copy()
# Find contours based on Edges
(new, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
NumberPlateCnt = None #we currently have no Number plate contour

for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour
            [X, Y, W, H] = cv2.boundingRect(c)
            break

print([X, Y, W, H])
cv2.rectangle(image, (X, Y), (X + W, Y + H), (0,0,255), 2)
cv2.imshow('new img',image)
drop_object_img=image[Y : Y+H , X : X+W]
cv2.imshow('drop object img',drop_object_img)
print(NumberPlateCnt.dtype)

#--------------------------------------------------------------------------------------
# second step

new_image = drop_object_img
new_image = cv2.resize(new_image, (916, 268)) 

# img= cv2.imread(r'C:\Users\luan1\Desktop\FRNNP.png')
img=new_image.copy()
img2=img.copy()

black = np.zeros_like(img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret2, th2 = cv2.threshold(gray, 0, 255 ,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow('threshold image',th2)

# print(ret2)

# thresh =ret2 
# im_bw = cv2.threshold(th2, thresh, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow('black to white',im_bw)
# th2=im_bw

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
th3=th2.copy()
threshed = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
cv2.imshow('morphology image',th3)

imgContours, Contours, Hierarchy = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE,offset=(0, 0))
array=[]
# print(Contours)
for contour in Contours:

    #--- select contours above a certain area ---
    if  1500 <cv2.contourArea(contour) and cv2.contourArea(contour) < 20000 :

        #--- store the coordinates of the bounding boxes ---
        [X, Y, W, H] = cv2.boundingRect(contour)
        array = np.append(array,[X,Y,W,H])
        array1 = np.reshape(array, (-1, 4))
        # array=np.concatenate(array,[X,Y,W,H])
        # print(array.shape)
        #--- draw those bounding boxes in the actual image as well as the plain blank image ---
        cv2.rectangle(img2, (X, Y), (X + W, Y + H), (0,0,255), 2)
        cv2.rectangle(black, (X, Y), (X + W, Y + H), (0,255,0), 2)
       
        # print(array[0])
        # cv2.imshow('roi 0',roi)#
        # roi = img[np_df[1]: np_df[1] + np_df[3], np_df[0] : np_df[0] + np_df[2]]
        # cv2.imshow('roi 0',roi)
cv2.imshow('contour', img2)
cv2.imshow('black', black)
# print(array)
# print(array1)
array2=sorted(array1, key=itemgetter(0))
array2=np.array(array2)
print(array2)
print(len(array2))
print(array2[0][1])
# roi=np.array([])
testtime = 1
for i in range(len(array2)):
    roi = threshed[int(array2[i][1]): int(array2[i][1]) + int(array2[i][3]), int(array2[i][0]) : int(array2[i][0]) + int(array2[i][2])]
    path = r'C:\Users\luan1\Desktop\New folder'
    cv2.imwrite(os.path.join(path , 'img{0}_{1}.jpg').format(testtime,i),roi)
    cv2.imshow('roi 1',roi)
else:
    testtime += 1 

text = pytesseract.image_to_string(threshed)

cv2.waitKey(0)