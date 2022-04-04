import numpy as np
import statistics as stat

def cal_ratio(rand_sample_list,class_list,classNum):
    # define a fisher ratio list for all columns with default value 0
    fisherRatioDic = {}
    colObjectList = []
    # for each column sample type we calculate one fisher ratio for one column
    rand_sample_list = np.array(rand_sample_list)
    for i in range(len(rand_sample_list[0])):
        #define a data list for all class
        all_data = rand_sample_list[:, i]
        class_data = []
        for j in range(classNum+1):
            class_data.append([])
        for z in range(len(class_list)):
            class_data[int(class_list[z])].append(all_data[z])
        #get rid of empty class
        class_data = [x for x in class_data if x != []]
        # Here we calculate the fisher ratio for that column
        # calculate the first lumda sqr
        all_data_mean = stat.mean(all_data)
        lumdaTop1 = 0
        for z in range(len(class_data)):
            class_data_mean = stat.mean(class_data[z])
            lumdaTop1 = lumdaTop1 + (((class_data_mean - all_data_mean)**2)*len(class_data[z]))
        lumdaBottom1 = classNum-1
        lumda1 = lumdaTop1/lumdaBottom1

        lumdaTop2_1 = 0
        for n in range(len(class_data)):
            for j in class_data[n]:
                lumdaTop2_1 = lumdaTop2_1 + (j - all_data_mean)**2

        lumdaTop2_2 = 0
        for p in range(len(class_data)):
            class_data_mean = stat.mean(class_data[p])
            lumdaTop2_2 = lumdaTop2_2 + (((class_data_mean - all_data_mean) ** 2) * len(class_data[p]))
        lumdaBottom2 = len(all_data) - classNum
        lumda2 = (lumdaTop2_1-lumdaTop2_2)/lumdaBottom2
        fisher_ratio = lumda1/lumda2
        fisherRatioDic[i] = fisher_ratio
    return fisherRatioDic