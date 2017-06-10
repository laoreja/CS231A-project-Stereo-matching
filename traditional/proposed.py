#!/usr/bin/python
import numpy as np
import math
import scipy.misc
#import os, sys
#import multiprocessing # failed
import threading
import pprint

dispMax = 228
imageW = 1226
imageH = 370

left = np.memmap('originOutput/leftCbca1.bin', dtype=np.float32, mode='c', shape=(1, dispMax, imageH, imageW))
right = np.memmap('originOutput/rightCbca1.bin', dtype=np.float32, mode='c', shape=(1, dispMax, imageH, imageW))


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

DEL = 0.2 # Python is case sensitive
INS = 0.2
W = imageW

def computeSUB(matchScore):
    if matchScore < 0.5:
        return 1.0 - matchScore
    else:
        return 0.0

def NWScoreLeft(vol_plane, idx):
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
#    return disp
    newDispLeft[idx] = disp.copy()
    
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


newDispLeft = np.zeros((imageH, imageW), dtype=np.float64)
newDispRight = np.zeros((imageH, imageW), dtype=np.float64)


num_threads = 5
for y in xrange(0, imageH, num_threads):
    threads = [threading.Thread(target=NWScoreRight, args=(right[0, :, y + i, :], y + i)) for i in range(num_threads) if y + i < imageH]
    threads.extend([threading.Thread(target=NWScoreLeft, args=(left[0, :, y + i, :], y + i)) for i in range(num_threads) if y + i < imageH])
    for t in threads:
        t.setDaemon(True)
        t.start()
    t.join()
    print y, 'done'
    
#for y in xrange(imageH):
##    newDispLeft[y] = NWScoreLeft(imageW, left[0, :, y, :])
#    newDispRight[y] = NWScoreRight(imageW, right[0, :, y, :])
#    print y, 'done'

leftCbcaSequenceBin = np.memmap('myOutput/leftCbcaSequence2.bin', dtype='float32', mode='w+', shape=newDispLeft.shape)
rightCbcaSequenceBin = np.memmap('myOutput/rightCbcaSequence2.bin', dtype='float32', mode='w+', shape=newDispRight.shape)

newDispLeft /= 1.0 * dispMax
newDispRight /= 1.0 * dispMax
scipy.misc.imsave('myOutput/leftCbcaSequenceDispLeft2.png', newDispLeft)
scipy.misc.imsave('myOutput/rightCbcaSequenceDispRight2.png', newDispRight)
# fill occ with 0
    

    
    
        
        
    
    


    

            

        