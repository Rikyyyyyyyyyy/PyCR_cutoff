import operator
import fisherRatio_in
import numpy as np
from sklearn.model_selection import train_test_split
import vipScore
import SelectivityRatio

def setNumber(classNum, classList, allSampleList, startNum,howMuchSplit,vRanking,nComponent):
    allSampleList = np.array(allSampleList)
    #get the half randomly selected sample and calculate the fisher ration
    sample_training, sample_test, class_training, class_test = selectRandom(allSampleList, classList,howMuchSplit)
    if vRanking == 'fisher':
        Ratio = fisherRatio_in.cal_ratio(sample_training, class_training, classNum)
    if vRanking == 'vip':
        Ratio = vipScore.vipy(sample_training, class_training, nComponent)
    if vRanking == 'selectivity':
        Ratio = SelectivityRatio.selrpy(sample_training, class_training, nComponent)
    sorted_Ratio = sorted(Ratio.items(), key=operator.itemgetter(1), reverse=True)

    # get the start variable list and end variable list by startNum and end Num
    startNumList = []
    for i in sorted_Ratio:
        if i[1] >= startNum:
            startNumList.append(i[0])
    return sample_training, startNumList

# randomly select half variables from the selected_scaled_variables_list
def selectRandom(sample_list,class_list, howMuchSplit):
    sample_matrix = np.array(sample_list)
    class_matrix = np.array(class_list)
    X_train, X_test, y_train, y_test = train_test_split(sample_matrix, class_matrix, test_size=float(howMuchSplit), stratify=class_matrix)
    return X_train, X_test, y_train, y_test
