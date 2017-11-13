import numpy as np
import cv2
import random
from PIL import Image
import os, os.path

oripath = "./s_oridata/"
noblurpath = "./s_cnn/train/no_blur/"
blurpath = "./s_cnn/train/blur/"
allimgpath = "./s_cnn/train/inputdata/"

size = 15
gridx=30
gridy=30

kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size

# img = Image.open(oripath+"JR.jpg")


#go through every image in source folder
print('begin loading images')
pi_imgs = []
cv_imgs = []
valid_images = [".jpg"]
for f in os.listdir(oripath):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    pi_imgs.append(Image.open(os.path.join(oripath,f)))
    cv_imgs.append(cv2.imread(os.path.join(oripath,f)))
print('finished loading images')
#

#
# looping to create blurry and non-blurry images in 50% chance
for i in range(len(pi_imgs)):
    img = pi_imgs[i]
    (imageWidth, imageHeight) = img.size

    rangex = imageWidth / gridx
    rangey = imageHeight / gridy
    for x in xrange(rangex):
        for y in xrange(rangey):

            bbox = (x * gridx, y * gridy, x * gridx + gridx, y * gridy + gridy)
            slice_bit = img.crop(bbox)
            if random.randrange(2) == 0:
                slice_bit.save(noblurpath + str(i)+'_'+ str(x) + '_' + str(y) + ',noblur.jpg', optimize=True, bits=6)
                slice_bit.save(allimgpath + str(i)+'_'+ str(x) + '_' + str(y) + ',noblur.jpg', optimize=True,bits=6)
                print(str(i))
            else:
                slice_bit.save(blurpath + str(i)+'_'+ str(x) + '_' + str(y) + ',blur.jpg', optimize=True, bits=6)
                img1 = cv2.imread(blurpath + str(i)+'_'+ str(x) + '_' + str(y) + ',blur.jpg')
                output = cv2.filter2D(img1, -1, kernel_motion_blur)
                cv2.imwrite(blurpath + str(i)+'_'+ str(x) + '_' + str(y) + ',blur.jpg', output)
                cv2.imwrite(allimgpath + str(i)+'_'+ str(x) + '_' + str(y) + ',blur.jpg', output)
                print(str(i))






