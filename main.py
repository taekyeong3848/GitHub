import tensorflow as tf
import numpy as np
import random
import math
from collections import defaultdict
import sys
import time

import dataprocess
import evaluation
import trainer

benchmark = ["ML100k"]                                  #be
dataType = ["d1","d2","d3","d4","d5"]                   #d
model = ["autorec", "apr", "dualTanh", "dualSigmoid", "bpr"]   #m
#model = ["autorec"] #"svd", "bpr", "svdpp"
SVD_model = ["svd", "svdpp"]

problem = ["explicit", "implicit"]                      #p
sampleFlag = [True, False]                              #s
stacked = [False, 2, 3, 4, 5]                           #st
denoising = [True, False]                               #de
topN = [5, 10, 15, 20]          

regularization = [0.1, 0.05, 0.01]                      #r  
hiddenLayer = [20, 40, 60, 80, 100, 120]                #h
optimizerType = [1, 2, 3] #1은 GD 2는 RMSprop, 3은 adam #o
learningRate = [0.1, 0.01, 0.001]    #l
rankingparam = [1, 1.25, 1.5, 1.75, 2]                  #ran
alpha = [1, 10, 20, 30, 40, 50]                            #a
beta = [0, 0.5, 0.4, 0.3, 0.6, 0.7]                        #b

sampler = [1] #1: full usage, 2: 특정 숫자만 이용
        
def learning (testData, testData_i, trainData, trainData_i, trainMask, trainMask_i, unratedItemsMask, positiveMask,
              negativeMask, numOfRatings, unrated_items, GroundTruth, GroundTruth_i, itemCount, m, h, reg, o, l, userCount):
    if m == "autorec":
        for p in problem:
            if p == "explicit":
                trainer.trainAutorec(testData, trainData, trainMask, unratedItemsMask, numOfRatings, unrated_items, GroundTruth, itemCount, topN, m, h, reg, o, l, p)
            elif p == "implicit":
                for s in sampleFlag:
                    trainer.trainAutorec(testData_i, trainData_i, trainMask_i, unratedItemsMask, numOfRatings, unrated_items, GroundTruth_i, itemCount, topN, m, h, reg, o, l, p, s)
                    
    elif m == "apr":
        for ran in rankingparam:
            for p in problem:
                if p == "explicit":
                    trainer.trainAPR(testData, trainData, trainMask, unratedItemsMask, positiveMask, negativeMask, numOfRatings, unrated_items, GroundTruth, itemCount, topN, m, h, reg, o, l, p, ran)
                elif p == "implicit":
                    trainer.trainAPR(testData_i, trainData_i, trainMask_i, unratedItemsMask, positiveMask, negativeMask, numOfRatings, unrated_items, GroundTruth_i, itemCount, topN, m, h, reg, o, l, p, ran)
                
    elif m == "dualTanh":
        for ran in rankingparam:
            for a in alpha:
                for b in beta:
                    trainer.trainDAT(testData, trainData, trainData_i, trainMask, trainMask_i, unratedItemsMask, numOfRatings, unrated_items, GroundTruth, itemCount, topN, m, h, reg, o, l, ran, a, b)
                    
    elif m == "dualSigmoid":
        for ran in rankingparam:
            for a in alpha:
                for b in beta:
                    trainer.trainDAS(testData, trainData, trainData_i, trainMask, trainMask_i, unratedItemsMask,
                                     positiveMask, numOfRatings, unrated_items, GroundTruth, itemCount, topN,
                                     m, h, reg, o, l, ran, a, b)
    
    elif m == "bpr":
        p = "implicit"
        trainer.trainBPR(testData, trainData, trainMask, unratedItemsMask, numOfRatings, unrated_items, GroundTruth,
                         itemCount, topN, m, h, reg, o, l, p)


    elif m == "svd":
        s = False
        p = "None"
        trainer.trainSVD(testData, trainData, trainMask, unratedItemsMask, numOfRatings, unrated_items, GroundTruth,
                         itemCount, topN, m, h, reg, o, l, p, userCount, s)
    elif m == "svdpp":
        s = False
        p = "None"
        trainer.trainSVDpp(testData, trainData, trainMask, unratedItemsMask, numOfRatings, unrated_items, GroundTruth,
                         itemCount, topN, m, h, reg, o, l, p, userCount, s)


# main program starts here                
for d in dataType:
    # prepare all the arrays need in training
    fileName = "../dataset/ML100k/" + d + ".train"
    userCount, itemCount, trainSet, testSet = dataprocess.loadData(fileName, d)

    print(userCount, itemCount)
    testData, testData_i, trainData, trainData_i, trainMask, trainMask_i, unratedItemsMask, positiveMask, negativeMask, numOfRatings, unrated_items, GroundTruth, GroundTruth_i = dataprocess.processData(itemCount, trainSet, testSet)
    # start training

    for m in model:
        TOTAL = len(regularization)*len(hiddenLayer)*len(optimizerType)*len(learningRate)
        COUNT = 1        
        
        for n in topN:
            outputFile = m + "_top" + str(n) + ".txt"
            f = open(outputFile, 'a')
            f.write("be d m p s st de r h o l ran a b precision recall ndcg mrr globalmrr cost epoch LAprecision LArecall LAndcg LAmrr LAglobalmrr\n")
            f.close()
        for h in hiddenLayer:
            for reg in regularization:
                for o in optimizerType:
                    for l in learningRate:
                        t1 = time.time()
                        print(m+" with %d/%d th process......" % (COUNT, TOTAL))
                        
                        # learn this model
                        learning(testData, testData_i, trainData, trainData_i, trainMask, trainMask_i, unratedItemsMask, positiveMask,
                                 negativeMask, numOfRatings, unrated_items, GroundTruth, GroundTruth_i,
                                 itemCount, m, h, reg, o, l, userCount)
                        
                        t2 = time.time()
                        print(m+" with %d/%d th process complete within %s seconds......" % (COUNT, TOTAL, str(t2-t1)))
                        COUNT = COUNT + 1

    for m in SVD_model:
        TOTAL = len(regularization) * len(optimizerType) * len(learningRate)
        COUNT = 1
        for n in topN:
            outputFile = m + "_top" + str(n) + ".txt"
            f = open(outputFile, 'a')
            f.write(
                "be d m p s st de r h o l ran a b precision recall ndcg mrr globalmrr cost epoch LAprecision LArecall LAndcg LAmrr LAglobalmrr\n")
            f.close()
        #time.sleep(10)
        for reg in regularization:
            for o in optimizerType:
                for l in learningRate:
                    t1 = time.time()
                    h = 0
                    print(m + " with %d/%d th process......" % (COUNT, TOTAL))

                    # learn this model
                    learning(testData, testData_i, trainData, trainData_i, trainMask, trainMask_i, unratedItemsMask,
                             positiveMask, negativeMask, numOfRatings, unrated_items, GroundTruth, GroundTruth_i, itemCount,
                             m, h, reg, o, l, userCount)

                    t2 = time.time()
                    print(m + " with %d/%d th process complete within %s seconds......" % (COUNT, TOTAL, str(t2 - t1)))
                    COUNT = COUNT + 1




                        

 

