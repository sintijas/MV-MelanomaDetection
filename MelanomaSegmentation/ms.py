import cv2
path = "ISIC_0000276.jpg"
melanoma = cv2.imread(path)
import numpy as np
import os

def centeredCrop(img):
   width =  np.size(img,1)
   height =  np.size(img,0)
   left = np.ceil((width - width/2)/2)
   left = int(left)
   top = np.ceil((height - height/2)/2)
   top = int(top)
   right = np.floor((width + width/2)/2)
   right = int(right)
   bottom = np.floor((height + height/2)/2)
   bottom = int(bottom)
   cImg = img[top:bottom, left:right]
   return cImg

def RemoveBackground(Image):
    IGray=cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
    ret,BW=cv2.threshold(IGray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret1,thresh1 = cv2.threshold(th,127,255,cv2.THRESH_BINARY)
    ret2,thresh2 = cv2.threshold(BW,127,255,cv2.THRESH_BINARY_INV)
    maskedImage=cv2.bitwise_and(Image,Image,mask=thresh2)
    return maskedImage,IGray

for image in os.listdir('Images'):
    img = cv2.imread('Images/'+image)
    img = centeredCrop(img)
    img = cv2.resize(img, (320,320))
    cv2.imwrite("ResizedImages/"+image, img)
    BGImage, A = RemoveBackground(img)
    cv2.imwrite('SegmentedImages/'+image, BGImage)



