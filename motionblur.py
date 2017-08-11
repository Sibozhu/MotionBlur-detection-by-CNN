import cv2
import numpy as np

img = cv2.imread('/Users/sibozhu/DeepLearning/testing/JR.jpg')


size = 15

# generating the kernel
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size

# applying the kernel to the input image
output = cv2.filter2D(img, -1, kernel_motion_blur)


cv2.imwrite('/Users/sibozhu/DeepLearning/testing/JR_blur.jpg',output)

