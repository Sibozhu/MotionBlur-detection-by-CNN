from PIL import Image
img = Image.open("/Users/sibozhu/DeepLearning/testing/JR_blur.jpg")
print(img)
(imageWidth, imageHeight)=img.size
gridx=60
gridy=60
rangex=img.width/gridx
rangey=img.height/gridy
print rangex*rangey
for x in xrange(rangex):
    for y in xrange(rangey):
        bbox=(x*gridx, y*gridy, x*gridx+gridx, y*gridy+gridy)
        slice_bit=img.crop(bbox)
        slice_bit.save('/Users/sibozhu/DeepLearning/testing/ppt/'+str(x)+','+str(y)+'.jpg', optimize=True, bits=6)
