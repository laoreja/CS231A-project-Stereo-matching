#!/usr/bin/python
import numpy as np
import math
import scipy.misc
#import os, sys
#import multiprocessing # failed
import threading
import pprint
from samples import bin2png

dispMax = 228
imageW = 1226
imageH = 370

leftImgName = 'samples/input/kittiL.png'
rightImgName = 'samples/input/kittiR.png'

leftImg = scipy.misc.imread(leftImgName, flatten=True).astype(np.float32)
leftImg /= 255.0

rightImg = scipy.misc.imread(rightImgName, flatten=True).astype(np.float32)
rightImg /= 255.0



leftCnn = np.memmap('myOutput/left.bin', dtype=np.float32, mode='c', shape=(1, dispMax, imageH, imageW))
rightCnn = np.memmap('myOutput/right.bin', dtype=np.float32, mode='c', shape=(1, dispMax, imageH, imageW))
#disp = np.memmap('myOutput/disp.bin', dtype=np.float32, mode='c', shape=(1, 1, imageH, imageW))

#print np.where(np.logical_not(np.isnan(left)))
#for t in np.where(np.isnan(left)):
#    print t.shape
#    print np.unique(t)


#left = left.copy()
#right = right.copy()
#
#print left.shape, right.shape
#
#left_min = np.min(left[np.where(np.logical_not(np.isnan(left)))])
#left_max = np.max(left[np.where(np.logical_not(np.isnan(left)))])
#
#right_min = np.min(right[np.where(np.logical_not(np.isnan(right)))])
#right_max = np.max(right[np.where(np.logical_not(np.isnan(right)))])
#
#print left_min, left_max
#print right_min, right_max

cbca_intensity = 0.13
cbca_distance = 5
cbca_iterations_1 = 2
cbca_iterations_2 = 0
# img need to be float-valued
def cbca(x, y, d, leftImg, rightImg, scoreMap, resMap):
    intensity_o_y_l = leftImg[y, x]
    intensity_o_y_r = rightImg[y, x]    
    
    cnt = 0
    costSum = 0
    
    for direction_y in [-1, 1]:
        y_delta = direction_y
        while abs(y_delta) < cbca_distance and y + y_delta >= 0 and y + y_delta < imageH and math.fabs(leftImg[y + y_delta, x] - intensity_o_y_l) < cbca_intensity and math.fabs(rightImg[y + y_delta, x] - intensity_o_y_r) < cbca_intensity:
            intensity_o_x_l = leftImg[y + y_delta, x]
            intensity_o_x_r = rightImg[y + y_delta, x - d]
            
            for direction_x in [-1, 1]:
                x_delta = direction_x
                while abs(x_delta) < cbca_distance and x - d + x_delta >= 0 and x + x_delta < imageW and math.fabs(leftImg[y + y_delta, x + x_delta] - intensity_o_x_l) < cbca_intensity and math.fabs(rightImg[y + y_delta, x - d + x_delta] - intensity_o_x_r) < cbca_intensity:
                    
                    if scoreMap[d, y + y_delta, x + x_delta] != float('nan'):
                        cnt += 1
                        costSum += scoreMap[d, y + y_delta, x + x_delta]
                        
                    x_delta += direction_x
            
            
            y_delta += direction_y
    if cnt > 0:
        costSum /= (1.0 * cnt)
        resMap[d, y, x] = costSum
#        return costSum
    else:
        resMap[d, y, x] = float('nan')
#        return float('nan')
    
def computeCbcaMap(leftImg, rightImg, costMap):
    num_threads = 11
    
    newCostMap = np.zeros_like(costMap)
    newCostMap.fill(float('nan'))
    
    for it in xrange(cbca_iterations_1):    
        for y in xrange(imageH):
            print 'y:', y
            for x in xrange(imageW):
                for d in xrange(0, min([dispMax, x]), num_threads):
                    threads = [threading.Thread(target=cbca, args=(x, y, d+i, leftImg, rightImg, costMap, newCostMap)) for i in range(num_threads) if d + i < min([dispMax, x])]
                    for t in threads:
                        t.setDaemon(True)
                        t.start()
                    t.join()
#                    newCostMap[d, y, x] = cbca(x, y, d, leftImg, rightImg, costMap)
        costMap = newCostMap.copy()
        newCostMap.fill(float('nan'))
    return costMap


leftCbca = computeCbcaMap(leftImg, rightImg, leftCnn[0])
print 'leftCbca.shape, ', leftCbca.shape
leftCbca= np.reshape(leftCbca, (1, dispMax, imageH, imageW))
leftCbcaBin = np.memmap('myOutput/leftCbcaMultiThreads.bin', dtype='float32', mode='w+', shape=leftCbca.shape)
leftCbcaPng = samples.getFigFromVol(leftCbcaBin)
scipy.misc.imsave('myOutput/leftCbcaMultiThreads.png', leftCbcaPng)
    

DEL = 0.1 # Python is case sensitive
INS = 0.1
W = imageW

def computeSUB(matchScore):
    if matchScore < 0.5:
        return 1.0 - matchScore
    else:
        return 0.0

def NWScoreLeft(vol_plane):
    score = np.zeros((W+1, W+1))
    path = np.empty((W+1, W+1), dtype=object)
    path.fill("n")
    for i in xrange(1, W + 1):
        score[i, 0] = score[i - 1, 0] + DEL
        path[i, 0] = "u"
        for j in xrange(max([1, i - dispMax + 1]), i + 1):
            matchScore = vol_plane[i - j, i - 1]
            try:
                assert(not math.isnan(matchScore))
            except AssertionError:
                print 'mathScore is NaN, i, j:', i, j
                exit(1)
            
            scoreSub = score[i - 1, j - 1] + computeSUB(matchScore)
            scoreIns = score[i, j - 1] + INS
            score[i, j] = max([scoreSub, scoreIns])
            if score[i, j] == scoreIns:
                path[i, j] = "l"
            elif score[i, j] == scoreSub:
                path[i, j] = "d"
            
            if i > j:
                scoreDel = score[i - 1, j] + DEL
                if scoreDel > score[i, j]:
                    score[i, j] = scoreDel
                    path[i, j] = "u"
    
    i = W
    j = W
    disp = np.zeros(W, dtype=np.float64)
#    disp.fill(0 / 0)
    while i > 0 or j > 0:
        if path[i, j] == "d":
            disp[i - 1] = i - j
            i -= 1
            j -= 1
        elif path[i, j] == "u":
            disp[i - 1] = 0
            i -= 1
        elif path[i, j] == "l":
            j -= 1
        else:
            print 'wrong path!'
            i = W
            j = W
            while path[i, j] != "n":
                print path[i, j], i, j
                if path[i, j] == "d":
                    i -= 1
                    j -= 1
                elif path[i, j] == "u":
                    i -= 1
                elif path[i, j] == "l":
                    j -= 1
                else:
                    pass
            exit(1)
    return disp
    
def NWScoreRight(vol_plane, idx):       
    score = np.zeros((W+1, W+1))
    path = np.empty((W+1, W+1), dtype=object)
    path.fill("n")
    for j in xrange(1, W + 1):
        score[0, j] = score[0, j - 1] + INS
        path[0, j] = "l"
    for i in xrange(1, W + 1):
        for j in xrange(i, min(W + 1, i + dispMax)):
            matchScore = vol_plane[j - i, i - 1]
            try:
                assert(not math.isnan(matchScore))
            except AssertionError:
                print 'mathScore is NaN, i, j:', i, j
                exit(1)
                
            scoreSub = score[i - 1, j - 1] + computeSUB(matchScore)            
            scoreDel = score[i - 1, j] + DEL
            score[i, j] = max([scoreSub, scoreDel])
            if score[i, j] == scoreDel:
                path[i, j] = "u"
            elif score[i, j] == scoreSub:
                path[i, j] = "d"
                
            if i < j:
                scoreIns = score[i, j - 1] + INS
                if scoreIns > score[i, j]:
                    score[i, j] = scoreIns
                    path[i, j] = "l"
    
    i = W
    j = W
    disp = np.zeros(W, dtype=np.float64)
#    disp.fill(0 / 0)
    while i > 0 or j > 0:
        if path[i, j] == "d":
            disp[i - 1] = j - i
            i -= 1
            j -= 1
        elif path[i, j] == "u":
            disp[i - 1] = 0
            i -= 1
        elif path[i, j] == "l":
            j -= 1
        else:
            print 'wrong path!'
            i = W
            j = W
            while path[i, j] != "n":
                print path[i, j], i, j
                if path[i, j] == "d":
                    i -= 1
                    j -= 1
                elif path[i, j] == "u":
                    i -= 1
                elif path[i, j] == "l":
                    j -= 1
            exit(1)
#    return disp
    newDispRight[idx] = disp.copy()


#newDispLeft = np.zeros((imageH, imageW), dtype=np.float64)
#newDispRight = np.zeros((imageH, imageW), dtype=np.float64)
#
#
#num_threads = 8
#for y in xrange(0, imageH, num_threads):
#    threads = [threading.Thread(target=NWScoreRight, args=(right[0, :, y + i, :], y + i)) for i in range(num_threads) if y + i < imageH]
#    for t in threads:
#        t.setDaemon(True)
#        t.start()
#    t.join()
#    print y, 'done'
    
#for y in xrange(imageH):
##    newDispLeft[y] = NWScoreLeft(imageW, left[0, :, y, :])
#    newDispRight[y] = NWScoreRight(imageW, right[0, :, y, :])
#    print y, 'done'
    
#newDispLeft /= 1.0 * dispMax
#newDispRight /= 1.0 * dispMax
#scipy.misc.imsave('myOutput/newDispLeft.png', newDispLeft)
#scipy.misc.imsave('myOutput/newDispRight.png', newDispRight)
# fill occ with 0
    

    
    
        
        
    
    


    

            

        