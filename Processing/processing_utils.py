'''
utility function package for image processing
by Sibo Zhu, Kieran Xiao Wang
2017.08.24
'''
import numpy as np
import cv2
import random
from PIL import Image
import PIL
from natsort import natsorted
import os, os.path

def save_image(image_np_array, image_save_path):
    '''
    !!! Sibo, Please complete this function !!!
    :param image_np_array: 3-D numpy array
    :param image_save_path: string
    :return:
    '''
    im = Image.fromarray(image_np_array)
    im.save(image_save_path)


def patch_merge_to_one_from_folder(patch_dir):
    '''
    merge patches into a whole image
    !!! Sibo, please complete this function !!!
    !!! now you are using naming for indicating patch location and blur/no blur, which is fine
    !!! the patch_dir is supposed to contain all patches of an image(no matter blured or not)
    !!! you may want to use os.walk
    !!! if so avoid to have any other .jpg file except image patches(e.g. do not save the whole image in .jpg in that folder)
    :param patch_dir: [string] directory to the folder that contains all patches(of an image)
    :return: [3-d array] whole image in np array
    '''
    patch_dir = patch_dir

    pi_imgs = []
    valid_images = [".jpg"]
    for f in os.listdir(patch_dir):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        pi_imgs.append(Image.open(os.path.join(patch_dir, f)))
    total = len(pi_imgs)

    print(str(len(pi_imgs)) + " patches in total")

    #######################################################
    """Loading all the images from directory"""
    dir = []
    valid_images = [".jpg"]
    for f in os.listdir(patch_dir):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        dir.append(os.path.join(patch_dir, f))

    #####################################################
    """concat images with numpy"""

    def concat_img_horizon(list_imgs):
        imgs = [PIL.Image.open(i) for i in list_imgs]
        # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        #
        imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
        imgs_comb = PIL.Image.fromarray(imgs_comb)
        return imgs_comb

    def concat_img_vertical(list_imgs):
        imgs = [PIL.Image.open(i) for i in list_imgs]
        # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
        # for a vertical stacking it is simple: use vstack
        imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
        imgs_comb = PIL.Image.fromarray(imgs_comb)
        return imgs_comb

    def concat_temp_horizon(list_imgs):
        imgs_comb = np.hstack((np.asarray(i) for i in list_imgs))
        imgs_comb = PIL.Image.fromarray(imgs_comb)
        return imgs_comb

    #########################################################


    """counting number of whole pictures in the folder"""
    max_index = 0
    for i in range(len(dir)):
        flag = int(dir[i].split("/")[-1].split(",")[0].split("_")[0])
        if flag > max_index:
            max_index = flag

    #############################################
    """placing patches to their certain picture"""
    pic_index = {}
    for elem in range(max_index + 1):
        pic_index[elem] = []

    for j in range(len(dir)):
        for k in range(len(pic_index)):
            if int(dir[j].split("/")[-1].split(",")[0].split("_")[0]) == k:
                pic_index[k].append(dir[j])

    print('the first picture contains ' + str(len(pic_index[0])) + ' patches')

    ##########################################
    """getting the total columns of picture"""

    def get_total_column(list):
        max_col = 0
        for l in range(len(list)):
            col_flag = int(pic_index[0][l].split("/")[-1].split(",")[0].split("_")[1])
            if col_flag > max_col:
                max_col = col_flag
        return max_col

    print ("this picture's total column is " + str(get_total_column(pic_index[0])))
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
        img = PIL.Image.open(res).convert("L")
        arr = np.array(img)
    return arr

def mass_patch_merge_to_one_from_folder(patch_dir,save_dir):
    '''
    After implementing the merging patches back to a whole image,
    we can also do that same thing to a folder that contains several patches that
    come from different images and merge and save them back to those original images (with partially
    blurry) based on the naming habit of slicing images.
    :param patch_dir: [string] directory to the folder that contains all patches(of an image)
    :param save_dir: [string] directory to the folder that used to save all those merged images
    :return: This time there's no return
    '''
    patch_dir = patch_dir
    result_dir = save_dir

    pi_imgs = []
    valid_images = [".jpg"]
    for f in os.listdir(patch_dir):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        pi_imgs.append(Image.open(os.path.join(patch_dir, f)))
    total = len(pi_imgs)

    print(str(len(pi_imgs)) + " patches in total")

    #######################################################
    """Loading all the images from directory"""
    dir = []
    valid_images = [".jpg"]
    for f in os.listdir(patch_dir):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        dir.append(os.path.join(patch_dir, f))


    #####################################################
    """concat images with numpy"""

    def concat_img_horizon(list_imgs):
        imgs = [PIL.Image.open(i) for i in list_imgs]
        # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        #
        imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
        imgs_comb = PIL.Image.fromarray(imgs_comb)
        return imgs_comb

    def concat_img_vertical(list_imgs):
        imgs = [PIL.Image.open(i) for i in list_imgs]
        # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
        # for a vertical stacking it is simple: use vstack
        imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
        imgs_comb = PIL.Image.fromarray(imgs_comb)
        return imgs_comb

    def concat_temp_horizon(list_imgs):
        imgs_comb = np.hstack((np.asarray(i) for i in list_imgs))
        imgs_comb = PIL.Image.fromarray(imgs_comb)
        return imgs_comb

    #########################################################


    """counting number of whole pictures in the folder"""
    max_index = 0
    for i in range(len(dir)):
        flag = int(dir[i].split("/")[-1].split(",")[0].split("_")[0])
        if flag > max_index:
            max_index = flag

    #############################################
    """placing patches to their certain picture"""
    pic_index = {}
    for elem in range(max_index + 1):
        pic_index[elem] = []

    for j in range(len(dir)):
        for k in range(len(pic_index)):
            if int(dir[j].split("/")[-1].split(",")[0].split("_")[0]) == k:
                pic_index[k].append(dir[j])

    print('the first picture contains ' + str(len(pic_index[0])) + ' patches')

    ##########################################
    """getting the total columns of picture"""

    def get_total_column(list):
        max_col = 0
        for l in range(len(list)):
            col_flag = int(pic_index[0][l].split("/")[-1].split(",")[0].split("_")[1])
            if col_flag > max_col:
                max_col = col_flag
        return max_col

    print ("this picture's total column is " + str(get_total_column(pic_index[0])))
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
        res.save(result_dir + str(a) + '.jpg')



def image_to_patch(image_path, patch_size, patch_dir):
    '''
    cut an image into patch with certain size
    !!! Sibo, please complete this function !!!

    :param image_path: [string] path to the image(e.g. ./whole_image.jpg)
    :param patch_size: [tuple] i.g. (30(length),30(witch))
    :param patch_dir: [string] dir where to save the patch dir.(patch dir is defined to be the folder that contains all
    image patches of an image)
    :return:
    '''
    img = Image.open(image_path)
    (imageWidth, imageHeight) = img.size
    gridx = patch_size
    gridy = patch_size
    rangex = img.width / gridx
    rangey = img.height / gridy
    print rangex * rangey
    for x in xrange(rangex):
        for y in xrange(rangey):
            bbox = (x * gridx, y * gridy, x * gridx + gridx, y * gridy + gridy)
            slice_bit = img.crop(bbox)
            slice_bit.save(patch_dir + str(x) + '_' + str(y) + '.jpg', optimize=True,
                           bits=6)
            print(patch_dir + str(x) + '_' + str(y) + '.jpg')
    print(imageWidth)


def directory_to_patch(patch_size,original_path,no_blur_path,blur_path,all_img_path):
    '''
    We take a directory that contains several whole pictures and cut then into custom size of patches,
    then apply 50% chance blur and non-blur to those patches, save them into blurry folder, non-blurry folder,
    and a folder that contains all blurry and non-blurry patches with order.
    :param patch_size: [integer] The custom patch size we want, e.g:for 30x30 patch, enter '30'
    :param original_path: [string] The original path that contains all the original pictures without any modification
    :param no_blur_path: [string] The destination path that contains all the non-blurry patches with order
    :param blur_path: [string] The destination path that contains all the blurry patches with order
    :param all_img_path: [string] The destination path that contains all the patches with order
    :return: There's no return in this function, all the modified patches are saved into the destination path
    '''
    #motion blur preset
    size = 15
    gridx = patch_size
    gridy = patch_size
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    # go through every image in source folder
    print('begin loading images')
    pi_imgs = []
    cv_imgs = []
    valid_images = [".jpg"]
    for f in os.listdir(original_path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        pi_imgs.append(Image.open(os.path.join(original_path, f)))
        cv_imgs.append(cv2.imread(os.path.join(original_path, f)))
    print('finished loading images')
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
                    slice_bit.save(no_blur_path + str(i) + '_' + str(x) + '_' + str(y) + ',noblur.jpg', optimize=True,
                                   bits=6)
                    slice_bit.save(all_img_path + str(i) + '_' + str(x) + '_' + str(y) + ',noblur.jpg', optimize=True,
                                   bits=6)
                    print(str(i))
                else:
                    slice_bit.save(blur_path + str(i) + '_' + str(x) + '_' + str(y) + ',blur.jpg', optimize=True, bits=6)
                    img1 = cv2.imread(blur_path + str(i) + '_' + str(x) + '_' + str(y) + ',blur.jpg')
                    output = cv2.filter2D(img1, -1, kernel_motion_blur)
                    cv2.imwrite(blur_path + str(i) + '_' + str(x) + '_' + str(y) + ',blur.jpg', output)
                    cv2.imwrite(all_img_path + str(i) + '_' + str(x) + '_' + str(y) + ',blur.jpg', output)
                    print(str(i))

