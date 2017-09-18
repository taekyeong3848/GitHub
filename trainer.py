import tensorflow as tf
import numpy as np
import random
import math
import dataprocess
import evaluation
import models
import time

tf.set_random_seed(123456789)
np.random.seed(123456789)
random.seed(123456789)

batchSize = 32
epochCount = 10 #epoch횟수 결정
DIM = 15

def initializeEvaluations(topN):
    bestPrecision = np.zeros(len(topN))
    bestRecall = np.zeros(len(topN))
    bestNDCG = np.zeros(len(topN))
    bestMRR = np.zeros(len(topN))

    bestCost = np.zeros(len(topN))
    bestEpochCount = np.zeros(len(topN))
    bestMeanForMeasures = np.zeros(len(topN))

    bestGlobalMRR = 0

    lastRecall = np.zeros(len(topN))
    lastPrecision = np.zeros(len(topN))
    lastNDCG = np.zeros(len(topN))
    lastMRR = np.zeros(len(topN))
    lastGlobalMRR = 0
    
    return bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR

# be d m p s st de r h o l ran a b precision recall ndcg mrr globalmrr cost epoch LAprecision LArecall LAndcg LAmrr LAglobalmrr
def printResultsToFile(topN, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount,
                       bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR, be, d, m, p, s, st, de, r, h, o, l, ran, a, b):
    for I in range(len(topN)):
        print("bestPreicison " + str(topN[I]) + ": %.4f" % bestPrecision[I])
        print("bestRecall " + str(topN[I]) + ": %.4f" % bestRecall[I])
        print("bestNDCG " + str(topN[I]) + ": %.4f" % bestNDCG[I])
        print("bestMRR " + str(topN[I]) + ": %.4f" % bestMRR[I])
        print("bestGlobalMRR: %.4f" % bestGlobalMRR)
        print("bestCost " + str(topN[I]) + ": %.4f" % bestCost[I])
        print("bestMean " + str(topN[I]) + ": %.4f" % bestMeanForMeasures[I])
        print("bestEpoch" + str(topN[I]) +": " + str(bestEpochCount[I]))
        print("")
        print("Last Preicison " + str(topN[I]) + ": %.4f" % lastPrecision[I])
        print("Last Recall " + str(topN[I]) + ": %.4f" % lastRecall[I])
        print("Last NDCG " + str(topN[I]) + ": %.4f" % lastNDCG[I])
        print("Last MRR " + str(topN[I]) + ": %.4f" % lastMRR[I])
        print("Last GlobalMRR : %.4f" % lastGlobalMRR)
        print("")
    
        outputFile = m + "_top" + str(topN[I]) + ".txt"
        f = open(outputFile, 'a')
        resultLine = (be+' '+d+' '+m+' '+p+' '+str(s)+' '+str(st)+' '+str(de)+' '+str(r)+' '+str(h)+' '
        +str(o)+' '+str(l)+' '+str(ran)+' '+str(a)+' '+str(b)+' '+(" %.4f" % bestPrecision[I])+' '+("%.4f" % bestRecall[I])+' ' +(" %.4f" % bestNDCG[I])+' '+("%.4f" % bestMRR[I])+' '+("%.4f" % bestGlobalMRR)+' '+("%.4f" % bestCost[I])+' '+("%.4f" % bestMeanForMeasures[I])+' '+str(bestEpochCount[I])+' '+(" %.4f" % lastPrecision[I])+' '+("%.4f" % lastRecall[I])+' '+(" %.4f" % lastNDCG[I])+' '+("%.4f" % lastMRR[I])+' '+("%.4f" % lastGlobalMRR)+'\n')
        f.write(resultLine)
        f.close()

        
# def trainBPR():


def trainSVD(testData, trainData, trainMask, unratedItemsMask, numOfRatings, unrated_items,
             GroundTruth, itemCount, topN, m, h, r, o, l, p, userCount, s = False):
    # prepare training
    bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, \
    lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR = initializeEvaluations(topN)
    userIdList_Train, allTrainData, allTestData, unratedTrainMask , test_ulist = dataprocess.prepareTrainAndTest(trainData,
                                                                                                    unratedItemsMask,
                                                                                                    testData)
    #print(userCount, itemCount)

    with tf.Graph().as_default():

        user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
        item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
        rate_batch = tf.placeholder(tf.float32, shape=[None])
        bias_user, infer, cost = models.SVD(user_batch, item_batch, rate_batch, user_num=userCount,
                                              item_num=itemCount, r=r, dim=DIM)

        if o == 1:
            optimizer = tf.train.GradientDescentOptimizer(l).minimize(cost)
        elif o == 2:
            optimizer = tf.train.RMSPropOptimizer(l).minimize(cost)
        else:
            optimizer = tf.train.AdamOptimizer(l).minimize(cost)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochCount):
            random.shuffle(userIdList_Train)

            for batchId in range(int(len(userIdList_Train) / batchSize)):
                start = batchId * batchSize
                end = start + batchSize
                batchUserId = []
                batchItemId = []
                batchRate = []
                for i in range(start, end):
                    userId = userIdList_Train[i]
                    for j in range(itemCount):
                        if(trainMask[userId][j] != 0):
                            batchUserId.append(userId)
                            batchItemId.append(j)
                            batchRate.append(trainData[userId][j])

            c, _ = sess.run([cost, optimizer], feed_dict= {user_batch: batchUserId,
                                                         item_batch: batchItemId,
                                                         rate_batch: batchRate})

            predictBatchRate = []
            #time.sleep(100)
            for u in test_ulist:
                predictBatchUserId = []
                predictBatchItemId = []
                for i in range(itemCount):
                    predictBatchUserId.append(u)
                    predictBatchItemId.append(i)
                rate = sess.run([infer], feed_dict={user_batch: predictBatchUserId,
                                                    item_batch: predictBatchItemId
                                                    })
                predictBatchRate.append(list(rate[0]))

            predictBatchRate = np.asarray(predictBatchRate)
            predictedValues, predictedIndices = sess.run(tf.nn.top_k(predictBatchRate * unratedTrainMask, itemCount))
            printTrigger, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, \
            bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR = \
                evaluation.evaluation_topN(GroundTruth, predictedIndices, topN, bestPrecision, bestRecall, bestNDCG,
                                           bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures,
                                           lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR, epoch, c)

            if printTrigger == True:
                print("%.4f   %.4f   %.4f   %.4f   %.4f" % (
                bestPrecision[0], bestRecall[0], bestNDCG[0], bestMRR[0], bestGlobalMRR))
            else:
                print("[" + p + " / sampling:" + str(s) + " Done...")

        sess.close()

        be = "ML100k"
        d = "d1"
        st = False
        de = False
        ran = 0
        a = 0
        b = 0
        # be d m p s st de r h o l ran a b precision recall ndcg mrr globalmrr cost epoch LAprecision LArecall LAndcg LAmrr LAglobalmrr
        printResultsToFile(topN, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount,
                           bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR, be, d, m,
                           p, s, st, de, r, h, o, l, ran, a, b)

# def trainSVDpp():


def trainAutorec(testData, trainData, trainMask, unratedItemsMask, numOfRatings, unrated_items, GroundTruth,
                 itemCount, topN, m, h, r, o, l, p, s=False):
    # prepare training
    bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, \
    lastPrecision, lastNDCG, lastMRR, lastGlobalMRR = initializeEvaluations(topN)
    userIdList_Train, allTrainData, allTestData, unratedTrainMask = dataprocess.prepareTrainAndTest(trainData, unratedItemsMask, testData)
    
    with tf.Graph().as_default():
        data, mask, y, cost = models.Autorec(itemCount, h, r)

        # define optimizer
        if o == 1:
            optimizer = tf.train.GradientDescentOptimizer(l).minimize(cost)
        elif o == 2:
            optimizer = tf.train.RMSPropOptimizer(l).minimize(cost)
        else:
            optimizer = tf.train.AdamOptimizer(l).minimize(cost)

        printTrigger = True
        # Start training
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochCount):
            random.shuffle(userIdList_Train)

            for batchId in range(int(len(userIdList_Train) / batchSize)):
                start = batchId * batchSize
                end = start + batchSize

                batchData = []
                batchMask = []
                
                for i in range(start, end):
                    userId = userIdList_Train[i]
                    batchData.append(trainData[userId])
                    if s:
                        # random negative sampling
                        unrated = np.zeros(itemCount)
                        tmp = np.random.choice(unrated_items[userId], numOfRatings[userId], replace=False)

                        for j in tmp:
                            unrated[j] = 1
                        batchMask.append(unrated + trainMask[userId])
                    else:
                        batchMask.append(trainMask[userId])
                batchData = np.array(batchData)
                batchMask = np.array(batchMask)

                c, _ = sess.run([cost, optimizer], feed_dict={data: batchData, mask: batchMask})

            print("[epoch %d/%d]\tcost : %.4f" % (epoch + 1, epochCount, c))

            userID_list = []
            for userId in testData:
                userID_list.append(userId)

            # calculate accuracy measures
            preidictedValues, predictedIndices = sess.run(tf.nn.top_k(y * mask, itemCount),
                                                             feed_dict={data: allTrainData,
                                                                        mask: unratedTrainMask})


            printTrigger, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, \
            bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR = \
                evaluation.evaluation_topN (GroundTruth, predictedIndices, topN, bestPrecision, bestRecall, bestNDCG,
                                            bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures,
                                            lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR, epoch, c)

            if printTrigger == True:
                print("%.4f   %.4f   %.4f   %.4f   %.4f" % (bestPrecision[0], bestRecall[0], bestNDCG[0], bestMRR[0], bestGlobalMRR))
            else:
                print("[" + p + " / sampling:" + str(s) + " Done...")
    
    sess.close()
    
    be="ML100k"
    d="d1"
    st=False
    de=False
    ran=0
    a=0
    b=0
    # be d m p s st de r h o l ran a b precision recall ndcg mrr globalmrr cost epoch LAprecision LArecall LAndcg LAmrr LAglobalmrr
    printResultsToFile(topN, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount,
                       bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR, be, d, m, p,
                       s, st, de, r, h, o, l, ran, a, b)
    
def trainAPR(testData, trainData, trainMask, unratedItemsMask, positiveMask, negativeMask, numOfRatings, unrated_items, GroundTruth, itemCount, topN, m, h, r, o, l, p, ran):

    bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR = initializeEvaluations(topN)
    with tf.Graph().as_default():
        data, mask, _positiveMask, _negativeMask, length_positive, length_negative, y, cost = models.APR(itemCount, h, r, ran)

        if o == 1:
            optimizer = tf.train.GradientDescentOptimizer(l).minimize(cost)
        elif o == 2:
            optimizer = tf.train.RMSPropOptimizer(l).minimize(cost)
        else:
            optimizer = tf.train.AdamOptimizer(l).minimize(cost)

        # prepare training
        userIdList_Train, allTrainData, allTestData, unratedTrainMask = dataprocess.prepareTrainAndTest(trainData, unratedItemsMask, testData)
        printTrigger = True
            
        # Start training
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochCount):
            random.shuffle(userIdList_Train)

            for batchId in range(int(len(userIdList_Train) / batchSize)):
                start = batchId * batchSize
                end = start + batchSize

                batchData = []
                batchMask = []
                batchPositiveMask = []
                batchNegativeMask = []
                len_positive = []
                len_negative = []
                
                for i in range(start, end):
                    userId = userIdList_Train[i]
                    batchData.append(trainData[userId])
                    batchMask.append(trainMask[userId])
                    batchPositiveMask.append(positiveMask[userId])
                    
                    # random negative sampling
                    unrated = np.zeros(itemCount)
                    tmp = np.random.choice(unrated_items[userId], numOfRatings[userId], replace=False)
                    for j in tmp:
                        unrated[j] = 1
                    batchNegativeMask.append(unrated)
                    
                    len_positive.append(numOfRatings[userId])
                    len_negative.append(numOfRatings[userId])
                    
                batchData = np.array(batchData)
                batchMask = np.array(batchMask)
                batchPositiveMask = np.array(batchPositiveMask)
                batchNegativeMask = np.array(batchNegativeMask)
                len_positive = np.array(len_positive)
                len_negative = np.array(len_negative)
                
                c, _ = sess.run([cost, optimizer], feed_dict={data: batchData, mask: batchMask, _positiveMask: batchPositiveMask, _negativeMask: batchNegativeMask, length_positive: len_positive, length_negative: len_negative})

            print("[epoch %d/%d]\tcost : %.4f" % (epoch + 1, epochCount, c))
         
            # calculate accuracy measures
            preidictedValues, predictedIndices = sess.run(tf.nn.top_k(y * mask, itemCount),
                                                             feed_dict={data: allTrainData,
                                                                        mask: unratedTrainMask})

            printTrigger, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR = evaluation.evaluation_topN (GroundTruth, predictedIndices, topN, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR, epoch, c)

            if printTrigger == True:
                print("%.4f   %.4f   %.4f   %.4f   %.4f" % (bestPrecision[0], bestRecall[0], bestNDCG[0], bestMRR[0], bestGlobalMRR))
            else:
                print("[" + p + " / rankingparam:" + str(ran) + " Done...")
    
    sess.close()
    # precision recall ndcg mrr globalmrr cost epoch LAprecision LArecall LAndcg LAmrr LAglobalmrr be d m p s st de r h o l ran a b 
    be="ML100k" 
    d="d1" 
    s=False
    st=False
    de=False
    a=0
    b=0
    printResultsToFile(topN, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR, be, d, m, p, s, st, de, r, h, o, l, ran, a, b)

def trainDAT(testData, trainData, trainData_i, trainMask, trainMask_i, unratedItemsMask, numOfRatings, unrated_items, GroundTruth, itemCount, topN, m, h, r, o, l, ran, a, b):

    bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR = initializeEvaluations(topN)
    with tf.Graph().as_default():
        data, mask, data_i, mask_i, length_positive, y, cost = models.DualAPR_tanh (itemCount, h, r, ran, a, b)
        
        if o == 1:
            optimizer = tf.train.GradientDescentOptimizer(l).minimize(cost)
        elif o == 2:
            optimizer = tf.train.RMSPropOptimizer(l).minimize(cost)
        else:
            optimizer = tf.train.AdamOptimizer(l).minimize(cost)

        # prepare training
        userIdList_Train, allTrainData, allTestData, unratedTrainMask = dataprocess.prepareTrainAndTest(trainData, unratedItemsMask, testData)
        printTrigger = True
            
        # Start training
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochCount):
            random.shuffle(userIdList_Train)

            for batchId in range(int(len(userIdList_Train) / batchSize)):
                start = batchId * batchSize
                end = start + batchSize

                batchData = []
                batchMask = []
                batchData_i = []
                batchMask_i = []
                len_positive = []
                
                for i in range(start, end):
                    userId = userIdList_Train[i]
                    batchData.append(trainData[userId])
                    batchMask.append(trainMask[userId])
                    batchData_i.append(trainData_i[userId])
                    batchMask_i.append(trainMask_i[userId])
                    len_positive.append(numOfRatings[userId])

                    
                batchData = np.array(batchData)
                batchMask = np.array(batchMask)
                batchData_i = np.array(batchData_i)
                batchMask_i = np.array(batchMask_i)
                len_positive = np.array(len_positive)
                
                c, _ = sess.run([cost, optimizer], feed_dict={data: batchData, mask: batchMask, data_i: batchData_i, mask_i: batchMask_i, length_positive: len_positive})

            print("[epoch %d/%d]\tcost : %.4f" % (epoch + 1, epochCount, c))
         
            # calculate accuracy measures
            preidictedValues, predictedIndices = sess.run(tf.nn.top_k(y * mask, itemCount),
                                                             feed_dict={data: allTrainData,
                                                                        mask: unratedTrainMask})

            printTrigger, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR = evaluation.evaluation_topN (GroundTruth, predictedIndices, topN, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR, epoch, c)

            if printTrigger == True:
                print("%.4f   %.4f   %.4f   %.4f   %.4f" % (bestPrecision[0], bestRecall[0], bestNDCG[0], bestMRR[0], bestGlobalMRR))
            else:
                print("[a: %d, b: %.2f, rankingparam: %.2f] Done..." % (a, b, ran))
    
    sess.close()
    # be d m p s st de r h o l ran a b precision recall ndcg mrr globalmrr cost epoch LAprecision LArecall LAndcg LAmrr LAglobalmrr
    be="ML100k"
    d="d1"
    p="None"
    s=False
    st=False
    de=False
    printResultsToFile(topN, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR, be, d, m, p, s, st, de, r, h, o, l, ran, a, b)

def trainDAS(testData, trainData, trainData_i, trainMask, trainMask_i, unratedItemsMask, positiveMask, numOfRatings, unrated_items, GroundTruth, itemCount, topN, m, h, r, o, l, ran, a, b):

    bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR = initializeEvaluations(topN)
    
    with tf.Graph().as_default():
        data, mask, data_i, mask_i, _positiveMask, length_positive, y, cost = models.DualAPR_sigmoid (itemCount, h, r, ran, a, b)
        
        if o == 1:
            optimizer = tf.train.GradientDescentOptimizer(l).minimize(cost)
        elif o == 2:
            optimizer = tf.train.RMSPropOptimizer(l).minimize(cost)
        else:
            optimizer = tf.train.AdamOptimizer(l).minimize(cost)

        # prepare training
        userIdList_Train, allTrainData, allTestData, unratedTrainMask = dataprocess.prepareTrainAndTest(trainData, unratedItemsMask, testData)
        printTrigger = True
            
        # Start training
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer()) #initialize_all_variables
        for epoch in range(epochCount):
            random.shuffle(userIdList_Train)

            for batchId in range(int(len(userIdList_Train) / batchSize)):
                start = batchId * batchSize
                end = start + batchSize

                batchData = []
                batchMask = []
                batchData_i = []
                batchMask_i = []
                batchPositiveMask = []
                len_positive = []
                
                for i in range(start, end):
                    userId = userIdList_Train[i]
                    batchData.append(trainData[userId])
                    batchMask.append(trainMask[userId])
                    batchData_i.append(trainData_i[userId])
                    batchMask_i.append(trainMask_i[userId])
                    batchPositiveMask.append(positiveMask[userId])
                    len_positive.append(numOfRatings[userId])

                batchData = np.array(batchData)
                batchMask = np.array(batchMask)
                batchData_i = np.array(batchData_i)
                batchMask_i = np.array(batchMask_i)
                batchPositiveMask = np.array(batchPositiveMask)
                len_positive = np.array(len_positive)
                
                c, _ = sess.run([cost, optimizer], feed_dict={data: batchData, mask: batchMask, data_i: batchData_i, mask_i: batchMask_i, _positiveMask: batchPositiveMask, length_positive: len_positive})

            print("[epoch %d/%d]\tcost : %.4f" % (epoch + 1, epochCount, c))
         
            # calculate accuracy measures
            preidictedValues, predictedIndices = sess.run(tf.nn.top_k(y * mask, itemCount),
                                                             feed_dict={data: allTrainData,
                                                                        mask: unratedTrainMask})

            printTrigger, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR = evaluation.evaluation_topN (GroundTruth, predictedIndices, topN, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR, epoch, c)

            if printTrigger == True:
                print("%.4f   %.4f   %.4f   %.4f   %.4f" % (bestPrecision[0], bestRecall[0], bestNDCG[0], bestMRR[0], bestGlobalMRR))
            else:
                print("[a: %d, b: %.2f, rankingparam: %.2f] Done..." % (a, b, ran))
    
    sess.close()
    # be d m p s st de r h o l ran a b precision recall ndcg mrr globalmrr cost epoch LAprecision LArecall LAndcg LAmrr LAglobalmrr
    be="ML100k"
    d="d1"
    p="None"
    s=False
    st=False
    de=False
    printResultsToFile(topN, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR, be, d, m, p, s, st, de, r, h, o, l, ran, a, b)