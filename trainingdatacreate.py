import numpy as np
import cv2
import random
from PIL import Image
import os, os.path

oripath = "/s_oridata/"
noblurpath = "/s_cnn/train/no_blur/"
blurpath = "/s_cnn/train/blur/"
#inputpath = "/Users/sibozhu/DeepLearning/testing/cnn/train/inputdata/"

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
# looping to create blured images
for i in range(len(pi_imgs)):
    # creating a blured copy of the original image
    img = pi_imgs[i]
    (imageWidth, imageHeight) = img.size
    rangex = img.width / gridx
    rangey = img.height / gridy
    for x in xrange(rangex):
        for y in xrange(rangey):

            bbox = (x * gridx, y * gridy, x * gridx + gridx, y * gridy + gridy)
            slice_bit = img.crop(bbox)
            if random.randrange(2) == 0:
                slice_bit.save(noblurpath + 'noblur,' +str(i)+'_'+ str(x) + '_' + str(y) + '.jpg', optimize=True, bits=6)
                #slice_bit.save(inputpath + 'noblur,' + str(i) + '_' + str(x) + '_' + str(y) + '.jpg', optimize=True,bits=6)
                print(str(i))
            else:
                slice_bit.save(blurpath + 'blur,' + str(i)+'_'+str(x) + '_' + str(y) + '.jpg', optimize=True, bits=6)
                img1 = cv2.imread(blurpath + 'blur,' +str(i)+'_'+ str(x) + '_' + str(y) + '.jpg')
                output = cv2.filter2D(img1, -1, kernel_motion_blur)
                cv2.imwrite(blurpath + 'blur,' +str(i)+'_'+ str(x) + '_' + str(y) + '.jpg', output)
                #cv2.imwrite(inputpath + 'blur,' + str(i) + '_' + str(x) + '_' + str(y) + '.jpg', output)
                print(str(i))






