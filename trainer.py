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
epochCount = 1 #epoch횟수 결정
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


def trainBPR(testData, trainData, trainMask, unratedItemsMask, numOfRatings, unrated_items, GroundTruth, itemCount, topN, m, h, reg, o, l,p, ran=False):
    bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR = initializeEvaluations(
        topN)
    #userIdList_Train, allTrainData, allTestData, unratedTrainMask = dataprocess.prepareTrainAndTest(trainData, unratedItemsMask, testData)


    user_count, item_count, user_ratings = dataprocess.load_bprdata_2(trainData)

    with tf.Graph().as_default(), tf.Session() as sess:
        u, i, j, mf_auc, bprloss, train_op, rating_mat = models.BPR(user_count, item_count, 20)
        r = 0.01


        user_ratings_test = generate_test(user_ratings)

        sess.run(tf.global_variables_initializer())
        for epoch in range(1, 11):
            _batch_bprloss = 0
            for k in range(1, 5000):  # uniform samples from training set
                uij = generate_train_batch(user_ratings, user_ratings_test, item_count)
                _bprloss, _ = sess.run([bprloss, train_op],
                                          feed_dict={u: uij[:, 0], i: uij[:, 1], j: uij[:, 2]})
                _batch_bprloss += _bprloss

            print("epoch: ", epoch)
            print("bpr_loss: ", _batch_bprloss / k)

            user_count = 0
            _auc_sum = 0.0

            # each batch will return only one user's auc
            for t_uij in generate_test_batch(user_ratings, user_ratings_test, item_count):
                _auc, _test_bprloss = sess.run([mf_auc, bprloss],
                                                  feed_dict={u: t_uij[:, 0], i: t_uij[:, 1], j: t_uij[:, 2]}
                                                  )
                user_count += 1
                _auc_sum += _auc
            print("test_loss: ", _test_bprloss, "test_auc: ", _auc_sum / user_count)
            print("")

            preidictedValues, predictedIndices = sess.run(tf.nn.top_k(rating_mat, itemCount),feed_dict={u: t_uij[:, 0], i: t_uij[:, 1], j: t_uij[:, 2]})
            indice_test = []
            #predictedIndices list(testData.keys())
            for k in testData.keys():
                indice_test.append(list(predictedIndices[k]))

            #indice_test =
            printTrigger, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR = evaluation.evaluation_topN(
                GroundTruth, indice_test, topN, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR,
                bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR,
                lastGlobalMRR, epoch, _batch_bprloss)
            if printTrigger == True:
                print("%.4f   %.4f   %.4f   %.4f   %.4f" % (
                bestPrecision[0], bestRecall[0], bestNDCG[0], bestMRR[0], bestGlobalMRR))
            else:
                print("[" + p + " / rankingparam:" + str(ran) + " Done...")

    sess.close()
    # precision recall ndcg mrr globalmrr cost epoch LAprecision LArecall LAndcg LAmrr LAglobalmrr be d m p s st de r h o l
    be = "ML100k"
    d = "d1"
    s = False
    st = False
    de = False
    ran = 0
    a = 0
    b = 0
    printResultsToFile(topN, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount,
                       bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR, be, d, m, p, s,
                       st, de, r, h, o, l)

def trainSVD(testData, trainData, trainMask, unratedItemsMask, numOfRatings, unrated_items,
             GroundTruth, itemCount, topN, m, h, r, o, l, p, userCount, s = False):
    # prepare training
    bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, \
    lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR = initializeEvaluations(topN)
    userIdList_Train, allTrainData, allTestData, unratedTrainMask, test_ulist = dataprocess.prepareTrainAndTest(trainData,
                                                                                                    unratedItemsMask,
                                                                                                    testData)
    #print(userCount, itemCount)

    with tf.Graph().as_default():

        user_batch, item_batch, rate_batch, infer, cost = models.SVD\
            (user_num=userCount,item_num=itemCount, r=r, dim=DIM)

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

            print("[epoch %d/%d]\tcost : %.4f" % (epoch + 1, epochCount, c))
            predictBatchRate = []
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

def trainSVDpp(testData, trainData, trainMask, unratedItemsMask, numOfRatings, unrated_items,
             GroundTruth, itemCount, topN, m, h, r, o, l, p, userCount, s = False):
    bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, \
    lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR = initializeEvaluations(topN)
    userIdList_Train, allTrainData, allTestData, unratedTrainMask, test_ulist = dataprocess.prepareTrainAndTest(
        trainData,unratedItemsMask,testData)
    # print(userCount, itemCount)

    with tf.Graph().as_default():
        user_batch, item_batch, rate_batch, rating_list_batch, userImplicit_batch,\
        embd_y, infer, cost = models.SVDpp(user_num=userCount, item_num=itemCount, r=r,
                                           dim=DIM)

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
                rated_itemsIndex = []
                numOfRatings_list = []
                userImplicit_list = []
                for i in range(start, end):
                    userId = userIdList_Train[i]
                    for j in range(itemCount):
                        if (trainMask[userId][j] != 0):
                            batchUserId.append(userId)
                            batchItemId.append(j)
                            batchRate.append(trainData[userId][j])
                            rated_itemsIndex.append(np.nonzero(trainData[userId]))
                            numOfRatings_list.append(numOfRatings[userId])

                for j in range(len(batchUserId)):
                    userImplicit = sess.run(embd_y, feed_dict={rating_list_batch:
                                                              rated_itemsIndex[j]})
                    userImplicit = np.reshape(userImplicit, (len(userImplicit[0]),
                                                                 len(userImplicit[0][0])))
                    userImplicit = np.sum(userImplicit, axis=0)
                    for k in range(DIM):
                        userImplicit[k] = (userImplicit[k]/math.sqrt(numOfRatings_list[j]))

                    userImplicit_list.append(userImplicit)

                c, _, predict = sess.run([cost, optimizer, infer], feed_dict={user_batch: batchUserId,
                                                              item_batch: batchItemId,
                                                              rate_batch: batchRate,
                                                              userImplicit_batch: userImplicit_list})


            print("[epoch %d/%d]\tcost : %.4f" % (epoch + 1, epochCount, c))

            predictRate = []
            for u in test_ulist:
                predictNumOfRate = numOfRatings[u]
                predictRateitemId = np.nonzero(testData[u])
                predictBatchUserId = []
                predictBatchItemId = []
                predictBatchUserImplicit = []
                for i in range(itemCount):
                    predictBatchUserId.append(u)
                    predictBatchItemId.append(i)

                tmp = sess.run(embd_y, feed_dict={rating_list_batch: predictRateitemId})
                tmp = np.reshape(tmp, (len(tmp[0]), len(tmp[0][0])))
                tmp = np.sum(tmp, axis=0)
                for k in range(DIM):
                    tmp[k] = (tmp[k] / math.sqrt(predictNumOfRate))
                for j in range(len(predictBatchUserId)):
                        predictBatchUserImplicit.append(tmp)

                rate = sess.run([infer], feed_dict={user_batch: predictBatchUserId,
                                                    item_batch: predictBatchItemId,
                                                    userImplicit_batch: predictBatchUserImplicit
                                                    })
                predictRate.append(list(rate[0]))

            predictRate = np.asarray(predictRate)
            predictedValues, predictedIndices = sess.run(tf.nn.top_k(predictRate * unratedTrainMask, itemCount))
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


def generate_test(user_ratings):
    '''
    for each user, random select one of his(her) rating into test set
    '''
    user_test = dict()
    for u, i_list in user_ratings.items():
        user_test[u] = random.sample(user_ratings[u], 1)[0]
    return user_test



def generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size=256):
    '''
    uniform sampling (user, item_rated, item_not_rated)
    '''
    t = []
    for b in range(batch_size):
        u = random.sample(user_ratings.keys(), 1)[0]
        i = random.sample(user_ratings[u], 1)[0]
        while i == user_ratings_test[u]:
            i = random.sample(user_ratings[u], 1)[0]

        j = random.randint(1, item_count)
        while j in user_ratings[u]:
            j = random.randint(1, item_count)
        t.append([u, i, j])
    return numpy.asarray(t)

def generate_test_batch(user_ratings, user_ratings_test, item_count):
    '''
    for an user u and an item i rated by u,
    generate pairs (u,i,j) for all item j which u has't rated
    it's convinent for computing AUC score for u
    '''
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        for j in range(1, item_count + 1):
            if not (j in user_ratings[u]):
                t.append([u, i, j])
        yield numpy.asarray(t)
