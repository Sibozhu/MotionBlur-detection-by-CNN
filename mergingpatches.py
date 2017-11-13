import numpy as np
import cv2
import random
from PIL import Image
import PIL
import os, os.path
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt

patch_dir = "./testing/"
temp_dir = "./testing/temp/"
result_dir = "./testing/result/"

pi_imgs = []
# cv_imgs = []
valid_images = [".jpg"]
for f in os.listdir(patch_dir):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    pi_imgs.append(Image.open(os.path.join(patch_dir,f)))
    # cv_imgs.append(cv2.imread(os.path.join(patch_dir,f)))
total = len(pi_imgs)

print(str(len(pi_imgs))+" patches in total")

#######################################################
"""Loading all the images from directory"""
dir=[]
valid_images = [".jpg"]
for f in os.listdir(patch_dir):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    dir.append(os.path.join(patch_dir,f))


##############################################################
'''
for testing here, no influence on global algorithm
'''


# flg = './testing/0_5_9,blur.jpg'
#
# im_index = flg.split("/")[-1].split(",")[0].split("_")[0]
# print("the image's index: "+str(im_index))
#
# im_width = flg.split("/")[-1].split(",")[0].split("_")[1]
# print("the image's width's code: "+str(im_width))
#
# im_height = flg.split("/")[-1].split(",")[0].split("_")[2]
# print("the image's height's code: "+str(im_height))

#######################################
"""concat images with numpy"""

def concat_img_horizon(list_imgs):
    imgs = [PIL.Image.open(i) for i in list_imgs]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    #
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    # imgs_comb.save( './testing/temp/'+str(save_name) +',horizon.jpg' )
    return imgs_comb

def concat_img_vertical(list_imgs):
    imgs = [PIL.Image.open(i) for i in list_imgs]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
    # for a vertical stacking it is simple: use vstack
    imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    # imgs_comb.save( './testing/temp/'+str(save_name) +',vertical.jpg' )
    return imgs_comb

def concat_temp_horizon(list_imgs):
    imgs_comb = np.hstack((np.asarray(i) for i in list_imgs))
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    return imgs_comb


#########################################################
'''
for testing here, no influence on global algorithm
'''
#
# list_im1 = pic_index[0][:3]
# list_im2 = pic_index[0][3:6]
# test1 = concat_img_vertical(list_im1)
# test2 = concat_img_vertical(list_im2)
# test3 = concat_temp_horizon(test1,test2)
# test4 = concat_temp_horizon(test2,test3)
# plt.imshow(test4)
# plt.show()

##############################################################

"""counting number of whole pictures in the folder"""
max_index=0
for i in range(len(dir)):
    flag = int(dir[i].split("/")[-1].split(",")[0].split("_")[0])
    if flag > max_index:
        max_index = flag
    # im_width = dir[i].split("/")[-1].split(",")[0].split("_")[1]
    # im_height = dir[i].split("/")[-1].split(",")[0].split("_")[2]

#############################################
"""placing patches to their certain picture"""
pic_index={}
for elem in range(max_index+1):
    pic_index[elem]=[]

for j in range(len(dir)):
    for k in range(len(pic_index)):
        if int(dir[j].split("/")[-1].split(",")[0].split("_")[0]) == k:
            pic_index[k].append(dir[j])

print('the first picture contains '+str(len(pic_index[0]))+' patches')

##########################################
"""getting the total columns of picture"""
def get_total_column(list):
    max_col = 0
    for l in range(len(list)):
        col_flag = int(pic_index[0][l].split("/")[-1].split(",")[0].split("_")[1])
        if col_flag > max_col:
            max_col = col_flag
    return max_col

print ("this picture's total column is "+str(get_total_column(pic_index[0])))
#########################################
"""getting the total rows of picture"""
def get_total_row(list):
    max_row = 0
    for o in range(len(list)):
        row_flag = int(list[o].split("/")[-1].split(",")[0].split("_")[2])
        if row_flag > max_row:
            max_row = row_flag
    return max_row
print ("this picture's total row is " + str(get_total_row(pic_index[0])))


########################################
"""sorting this picture's patches with order of name"""
def sort_picture_patches(list):
    return natsorted(list)
#####################################

"""doing global merging"""

for a in range(len(pic_index)):
    max_col = get_total_column(pic_index[a])
    max_row = get_total_row(pic_index[a])
    pic_index[a] = sort_picture_patches(pic_index[a])
    col_index = {}
    for elem in range(max_col + 1):
        col_index[elem] = []

    for j in range(len(pic_index[a])):
        for k in range(max_col):
            if int(pic_index[a][j].split("/")[-1].split(",")[0].split("_")[1]) == k:
                col_index[k].append(pic_index[a][j])

    saver = []

    for i in range(max_col - 1):
        flag = concat_img_vertical(col_index[i])
        saver.append(flag)

    res = concat_temp_horizon(saver)
    res.save(result_dir + str(a)+ '.jpg')

