import numpy as np
import cv2
import random
from PIL import Image
import os, os.path

#
# NOTE: this function creates new partly blured images to tarpath folder
#       and return a 3D-list labels which indicates blured parts in each image
#
# NOTE: please enter your correct folder path
# NOTE: transpath is just a temp path that stores the blured copies of images
# NOTE: Oripath is the source path and tarpath is where all the outputs are.
oripath = "./s_oridata/"
transpath = "./transimg/"
tarpath = "./tarimg/"

# global variables
size = 15

labels = []

gridx = 30
gridy = 30

kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size

# go through every image in source folder
pi_imgs = []
cv_imgs = []
valid_images = [".jpg"]
for f in os.listdir(oripath):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    pi_imgs.append(Image.open(os.path.join(oripath,f)))
    cv_imgs.append(cv2.imread(os.path.join(oripath,f)))


# looping to create blured images
for i in range(len(pi_imgs)):
    # creating a blured copy of the original image
    temp = cv_imgs[i]
    blured = cv2.filter2D(temp, -1, kernel_motion_blur)
    cv2.imwrite(transpath+str(i)+".jpg",blured)
    
    # reload both original and blured version
    img1 = pi_imgs[i]
    img2 = Image.open(transpath+str(i)+".jpg")
    
    # cut the image into 30*30 pieces 
    # and randomly choose half of the pieces to blur
    (imageWidth, imageHeight)=img1.size
    rangex=int(img1.width/gridx)
    rangey=int(img1.height/gridy)
    t_parts = rangex*rangey
    
    # creating labels that identifies which part is blured
    label = []
    for j in range(rangex):
        label += [[0]*rangey]
    
    
    for k in range(int(t_parts/2)):
        x = random.randint(0,rangex-1)
        y = random.randint(0,rangey-1)
        label[x][y] = 1
        
        box = (x*gridx,y*gridy,x*gridx+gridx,y*gridy+gridy)
        sliced = img2.crop(box)
            
    
        img1.paste(sliced, (x*gridx,y*gridy))
        
    img1.save(tarpath + str(i)+".jpg")
    
    labels += [label]



    
    
    
    
    
