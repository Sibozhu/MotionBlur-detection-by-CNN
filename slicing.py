from PIL import Image
save_path = "./testing/"

img = Image.open("/Users/sibozhu/DeepLearning/testing/JR.jpg")
(imageWidth, imageHeight)=img.size
gridx=30
gridy=30
rangex=imageWidth/gridx
print('Width: '+str(rangex))
rangey=imageHeight/gridy
print('Height: '+str(rangey))
print str(rangex*rangey) + " patches in total"
for x in xrange(rangex):
    for y in xrange(rangey):
        bbox=(x*gridx, y*gridy, x*gridx+gridx, y*gridy+gridy)
        slice_bit=img.crop(bbox)
        slice_bit.save(save_path+str(x)+'_'+str(y)+'.jpg', optimize=True, bits=6)
        # print (save_path+str(x)+'_'+str(y)+'.jpg')
print imageWidth

