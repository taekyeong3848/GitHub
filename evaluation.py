import numpy as np
import math

def ndcg_func(yk, iyk, k):
    ndcg=0

    dcg_i = 0
    idcg_i = 0

    for j in range(0, k):
        dcg_i += ((math.pow(2,yk[j])-1)/math.log2(j+2))
    for j in range(0, k):
        idcg_i += ((math.pow(2,iyk[j])-1)/math.log2(j+2))
    if(idcg_i!=0):
        ndcg += (dcg_i/idcg_i)
    return ndcg

def evaluation_topN (GroundTruth, predictedIndices, topN, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR, epoch, c):

    globalMRR = 0
    
    for i in range(len(predictedIndices)):
        userMRR = 0

        hit = np.intersect1d(GroundTruth[i], predictedIndices[i])
        for j in hit:
            userMRR += (1/(list(predictedIndices[i]).index(j)+1))    #인덱스분에 1의 합
        if len(hit) != 0:
            globalMRR += (userMRR / len(hit))
            
    globalMRR = globalMRR / len(predictedIndices)
    if bestGlobalMRR > globalMRR:
        pass
    else:
        bestGlobalMRR = globalMRR

    # precision recall ndcg mrr @ N
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg=0
        sumForMRR=0
        pred_indices = list([[j[i] for i in range(topN[index])] for j in predictedIndices])
        
        for i in range(len(pred_indices)):
            if len(GroundTruth[i]) != 0:
                hit = np.intersect1d(GroundTruth[i], pred_indices[i])
                ######## precision recall #######
                
                sumForPrecision += len(hit) / topN[index]
                sumForRecall += (len(hit) / len(GroundTruth[i]))
                    
                ######## ndcg #######
                y_k = []
                i_y_k = []
                for j in pred_indices[i]:
                    if j in GroundTruth[i]:
                        y_k.append(1)
                    else:
                        y_k.append(0)
                    i_y_k.append(1)
                sumForNdcg += ndcg_func(y_k, i_y_k, topN[index])
                
                ######## MRR #######
                userMRR = 0
                for j in hit:
                    userMRR += (1/(list(pred_indices[i]).index(j)+1))    #인덱스분에 1의 합
                if len(hit) != 0:
                    sumForMRR += (userMRR / len(hit))
        
        # save the best ones
        precision = sumForPrecision / len(pred_indices)
        recall = sumForRecall / len(pred_indices)
        mrr = sumForMRR / len(pred_indices)
        ndcg = sumForNdcg / len(pred_indices)
        meanForMeasures = (precision + recall + ndcg + mrr) / 4
        
        if (bestMeanForMeasures[index] > meanForMeasures):
            printTrigger = False
        else:
            printTrigger = True
            bestMeanForMeasures[index] = meanForMeasures
            bestPrecision[index] = precision
            bestRecall[index] = recall
            bestNDCG[index] = ndcg
            bestMRR[index] = mrr
            bestEpochCount[index] = (epoch + 1)
            bestCost[index] = c
            
        lastPrecision[index] = precision
        lastRecall[index] = recall
        lastNDCG[index] = ndcg
        lastMRR[index] = mrr
    lastGlobalMRR = globalMRR
    
    return printTrigger, bestPrecision, bestRecall, bestNDCG, bestMRR, bestGlobalMRR, bestCost, bestEpochCount, bestMeanForMeasures, lastRecall, lastPrecision, lastNDCG, lastMRR, lastGlobalMRR
