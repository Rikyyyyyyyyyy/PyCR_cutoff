import numpy as np
import random
import copy
from statistics import NormalDist
import matplotlib.pyplot as plt
from scipy.stats import norm
import vipScore_in
import SelectivityRatio_in
import scipy.stats
import math
from scipy import special
from scipy.stats import f
def gaussian_algorithm(classNum,class_list,valList,VRanking, nComponent):
    ITERATIONS = 100
    sample_matrix = np.array(valList)
    k = 0
    true_means = []
    while k < ITERATIONS:
        # get the half random sample and half random variables list
        half_rand_matrix, sample_ind_list = selectHalfRandom(sample_matrix)
        half_rand_class_list = []
        for z in sample_ind_list:
            half_rand_class_list.append(class_list[z])
        classNum_list = []
        for i in range(classNum):
            classNum_list.append(i + 1)
        # create true distribution with the original class number
        true_dist_classNum = copy.deepcopy(half_rand_class_list)
        # calculate the tru and null fisher ratio
        if VRanking == 'fisher':
            true_fisherRatio = cal_fish_ratio(half_rand_matrix, true_dist_classNum, classNum)
        elif VRanking == 'vip':
            true_fisherRatio = vipScore_in.vipy(half_rand_matrix, true_dist_classNum, nComponent)
        elif VRanking == 'selectivity':
            true_fisherRatio = SelectivityRatio_in.selrpy(half_rand_matrix, true_dist_classNum, nComponent)
        true_mean_fisher_ratio = np.mean(true_fisherRatio)
        true_means.append(true_mean_fisher_ratio)
        k = k + 1
    for i in range(len(true_means)):
        if true_means[i] == np.inf:
            true_means[i] = 0
    true_fisher_mean = np.mean(true_means)
    true_fisher_std = np.std(true_means)

    ####################################  START GRAPH CODE ###################################
    #replace the inf in mean with the largest number in original matrix
    max_range = max(true_means)
    # start the true and null gaussian
    true_n, true_bins, true_patches = plt.hist(true_means, density=True,color="#3468eb",alpha=.6, label="true fisher mean", range=(0,max_range),bins = 35)
    true_mean, true_std = norm.fit(true_means)
    # true_y = ((1 / (np.sqrt(2 * np.pi) * true_fisher_std)) * np.exp(-0.5 * (1 / true_fisher_std * (true_bins - true_fisher_mean)) ** 2))
    true_y = ((1 / (np.sqrt(2 * np.pi) * true_std)) * np.exp(
        -0.5 * (1 / true_std * (true_bins - true_mean)) ** 2))
    plt.plot(true_bins, true_y, '--', color="#3468eb")

    ## Calculate the overlapping between two normalization
    startNum = NormalDist(mu=true_fisher_mean, sigma=true_fisher_std).inv_cdf(0.95)
    ## Draw the graph

    plt.tight_layout()
    mn, mx = plt.xlim()
    plt.plot([startNum, startNum], [0, 0.5], color='blue', label="Start Number")
    plt.plot(startNum, 0.5, color='blue')
    plt.legend(loc='best')
    plt.xlim(mn, mx)
    plt.title("Start and stop number determination via CLT")
    if VRanking == 'fisher':
        plt.ylabel("Likelihood")
        plt.xlabel("Mean of Fisher ratio")
        plt.savefig('output/FisherMean.png', bbox_inches="tight")
        plt.figure().clear()
    if VRanking == 'vip':
        plt.ylabel("Likelihood")
        plt.xlabel("Mean of VIP ratio")
        plt.savefig('output/VIPmean.png', bbox_inches="tight")
        plt.figure().clear()
    if VRanking == 'selectivity':
        plt.ylabel("Likelihood")
        plt.xlabel("Mean of Selectivity ratio")
        plt.savefig('output/SelectivityMean.png', bbox_inches="tight")
        plt.figure().clear()
    return startNum

def cutoff_algorithm(classNum,class_list,valList,VRanking):
    sample_matrix = np.array(valList)
    half_rand_class_list = []
    half_rand_matrix, sample_ind_list = selectHalfRandom(sample_matrix)
    for z in sample_ind_list:
        half_rand_class_list.append(class_list[z])
    true_dist_classNum = copy.deepcopy(half_rand_class_list)
    if VRanking == 'vip':
        return 1
    elif VRanking =='fisher':
        true_fisherRatio = cal_fish_ratio(half_rand_matrix, true_dist_classNum, classNum)
        dfn = classNum-1
        dfd = len(sample_matrix) - classNum

        x = np.linspace(f.ppf(0.01, dfn, dfd),
                        f.ppf(0.99, dfn, dfd), 100)
        plt.plot(x,f.pdf(x, dfn, dfd),
                '--', color='#3468eb', lw=2, alpha=0.6, label='F pdf')
        plt.hist(true_fisherRatio, density=True, color="#3468eb", alpha=.95, label="observed F values",
                 range=(0, max(x)), bins=60)
        mn, mx = plt.xlim()
        y_min, y_max = plt.ylim()
        fisher_startNum = f.ppf(q=0.95, dfn=dfn, dfd=dfd)
        fisher_endNum = f.ppf(q=0.5, dfn=dfn, dfd=dfd)
        pos = y_max/2
        plt.plot([fisher_startNum, fisher_startNum], [0, pos], color='blue', label="Start Number")
        plt.plot([fisher_endNum, fisher_endNum], [0, pos], color='red', label="End Number")
        plt.plot(fisher_startNum, pos, color='blue',label="Start Number")
        plt.plot(fisher_endNum, pos, color='red', label="End Number")
        plt.legend(loc='best')
        plt.xlim(mn, mx)
        plt.title("Theoretical and observed distribution of F values")
        plt.savefig('output/Theoretical_and_observed_distribution_of_F_values.png', bbox_inches="tight")
        plt.figure().clear()
        return fisher_startNum
    elif VRanking == 'selectivity':
        sample_num = len(class_list)
        startNumber = scipy.stats.f.ppf(q=1-0.05, dfn=sample_num-2, dfd=sample_num-3)
        return startNumber



# randomly get half sample, half variable matrix
def selectHalfRandom(sample_list):
    idx_list = []
    rand_sample_list = []
    for i in range(len(sample_list)):
        idx_list.append(i)

    total_num = len(sample_list)
    half_num = total_num//2

    rand_idx_list = random.sample(list(idx_list),half_num)
    for idx in rand_idx_list:
        rand_sample_list.append(sample_list[idx])

    return rand_sample_list,rand_idx_list

def cal_fish_ratio(sample_list,class_list,classNum):
    # define a fisher ratio list for all columns with default value 0
    fish_ratio = []
    # for each column sample type we calculate one fisher ratio for one column
    for i in range(len(sample_list[0])):
        #define a data list for all class
        # define a data list contain different class data list
        class_data = []
        for k in range(classNum + 1):
            class_data.append([])
        #for each row of data
        all_data = [row[i] for row in sample_list]
        for ind in range(len(all_data)):
            class_data[int(class_list[ind])].append(all_data[ind])
        class_data = [x for x in class_data if x != []]
        # Here we calculate the fisher ratio for that column
        # calculate the first lumda sqr
        all_data_mean = np.mean(all_data)
        lumdaTop1 = 0
        for z in range(len(class_data)):
            class_data_mean = np.mean(class_data[z])
            lumdaTop1 = lumdaTop1 + (((class_data_mean - all_data_mean)**2)*len(class_data[z]))
        lumdaBottom1 = classNum-1
        lumda1 = lumdaTop1/lumdaBottom1
        lumdaTop2_1 = 0
        for n in range(len(class_data)):
            for j in class_data[n]:
                lumdaTop2_1 = lumdaTop2_1 + (j - all_data_mean)**2
        lumdaTop2_2 = 0
        for p in range(len(class_data)):
            class_data_mean = 0
            for data in class_data[p]:
                class_data_mean = class_data_mean + data
            class_data_mean = class_data_mean/len(class_data[p])
            lumdaTop2_2 = lumdaTop2_2 + (((class_data_mean - all_data_mean) ** 2) * len(class_data[p]))
        lumdaBottom2 = len(all_data) - classNum
        lumda2 = (lumdaTop2_1-lumdaTop2_2)/lumdaBottom2
        fisher_ratio = lumda1/lumda2
        fish_ratio.append(fisher_ratio)
    fish_ratio = np.nan_to_num(fish_ratio, nan=(10 ** -12))
    return fish_ratio