#!/usr/bin/python
import numpy as np
import png

# tested, compared with disp_read.m
def load_uint16PNG(filename):
  reader = png.Reader(filename)
  pngdata = reader.read()
  px_array = np.array( map( np.uint16, pngdata[2] ))
  
  mask = px_array != 0
  disparity = px_array.astype(np.float32) / 256.0

#    print px_array.dtype
#    print px_array.shape
#    print np.sum(mask)
#    print np.where(mask == True)
#    print mask[125:130, 873:880]
#    print disparity[125:130, 873:880]    
#    print px_array[125:130, 873:880]    
  return disparity, mask

# Though the output disparity looks similar, but the image size does not match, may related to the compression level
# So better use what is provided by the devkit
# Question is what is the type of D
def save_uint16PNG(filename, disparity):
  disparity = disparity.astype(np.float64)
  img = disparity * 256
  img[np.where(disparity == 0)] = 1
  img[np.where(img < 0)] = 0
  img[np.where(img > 65535)] = 0
  img = img.astype(np.uint16)
#    print img[125:130, 873:880]        

  with open(filename, 'wb') as fd:
    w = png.Writer(width = disparity.shape[1], height = disparity.shape[0], greyscale=True, bitdepth=16)
    w.write(fd, img)

if __name__ == '__main__':
  filename = '/Users/laoreja/study/CS231A/project/datasets/KITTI/devkit/disparity.png'
  d, m = load_uint16PNG(filename)
  
  save_uint16PNG('/Users/laoreja/study/CS231A/project/datasets/KITTI/devkit/disparity_write.png', d)
  