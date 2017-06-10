#!/usr/bin/python
import numpy as np
import scipy.misc

dispMax = 228
imageW = 1226
imageH = 370

left = np.memmap('originOutput/leftSgm.bin', dtype=np.float32, mode='c', shape=(1, dispMax, imageH, imageW))
right = np.memmap('originOutput/rightSgm.bin', dtype=np.float32, mode='c', shape=(1, dispMax, imageH, imageW))
disp = np.memmap('originOutput/dispSgm.bin', dtype=np.float32, mode='c', shape=(1, 1, imageH, imageW))


def getFigFromVol(input):
    output = input[0].copy()
    print output[:2, 100:103, 50:53]
    
    output = np.nanargmin(output, axis=0).astype(np.float32)
    print output[100:103, 50:53]

    min_ = np.min(output)
    max_ = np.max(output)
    print output.shape, min_, max_
    
    output = (output - min_) / ((max_ - min_) * 1.0)
    print output[100:103, 50:53]
    return output
    
leftOutput = getFigFromVol(left)
rightOutput = getFigFromVol(right)

disp_min = np.min(disp)
disp_max = np.max(disp)
print 'disp min, max:', disp_min, disp_max
dispOutput = (disp[0, 0] - disp_min) / (1.0 * (disp_max - disp_min))
print dispOutput.shape

scipy.misc.imsave('originOutput/leftSgmPython.png', leftOutput)
scipy.misc.imsave('originOutput/rightSgmPython.png', rightOutput)
scipy.misc.imsave('originOutput/dispSgmPython.png', dispOutput)