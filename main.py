import copy
import numpy as np
import newScore
import genStartEndNum2
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from scipy.stats.distributions import chi2
from sklearn.multiclass import OneVsRestClassifier
import sys
from scipy.sparse.linalg import svds
from numpy import inf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xlsxwriter
import math
from colour import Color
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import file_pkg

def main(isexternal,howMuchSplit,isMicro,tupaType,isMotabo,MotaboFileName,DataFileName,ClassFileName,sampleNameFile,variableNameFile,scale_type,iteration,survivalRate,V_rankingAlgorithm,nComponent,isCutoff):
    ITERATION = iteration
    # generate roc color
    red = Color("#dc3c40")
    ROC_COLOR = list(red.range_to(Color("#55a6bc"), ITERATION + 1))

    # class color
    CLASS_COLOR = ["#dc3c40", "#55a6bc", 'purple', 'yellowgreen', 'wheat', 'royalblue', '#42d7f5', '#ca7cf7', '#d2f77c']
    CLASS_LABEL = ["o", "x", "4", "*", "+", "D", "8", "s", "p"]
    print("Generating the output ...")

    # create the needed folder to save ouput data
    file_pkg.create_folder()
    if isMotabo:
        sampleList, sampleName, classList, variableName =file_pkg.readMotabo(MotaboFileName)
    else:
        classList = file_pkg.getValFromFileByCols(ClassFileName)[0]
        sampleList = file_pkg.getValFromFileByRows(DataFileName)
        sampleName = file_pkg.getValFromFileByCols(sampleNameFile)[0]
        variableName = file_pkg.getValFromFileByCols(variableNameFile)

    # Trans classList
    unique_class = set(classList)
    unique_class = sorted(list(unique_class))
    classNum = len(unique_class)
    # trans the class
    class_trans_dict = {}
    for i in range(classNum):
        class_trans_dict[unique_class[i]] = str(i+1)
    for key in class_trans_dict.keys():
        classList = [sub.replace(key, class_trans_dict[key]) for sub in classList]
    classList = [int(x) for x in classList]

    class_num_label = []
    for i in range(1, classNum + 1):
        class_num_label.append(i)
    # get the variable list

    ori_sample = np.array(sampleList)
    ori_class = classList

    #Do class Tupa
    if tupaType.lower() =='tupa':
        sampleList = tupa(sampleList,classList)
    elif tupaType.lower()=='classtupa':
        sampleList = class_tupa(sampleList, classList)
    hori_index = []
    indice_list =[]
    for i in range(len(sampleList[0])):
        hori_index.append(i)
    for j in range(len(classList)):
        indice_list.append(j)
    if tupaType.lower() == 'tupa':
        file_pkg.export_file(sampleList, ori_class, indice_list, hori_index, 'output/tupaAllSample.csv', class_trans_dict,sampleName,variableName)
    elif tupaType.lower() == 'classtupa':
        file_pkg.export_file(sampleList, ori_class, indice_list, hori_index, 'output/ClassTupaAllSample.csv', class_trans_dict,sampleName,variableName)
    file_pkg.export_file(ori_sample, ori_class, indice_list, hori_index, 'output/original_file.csv', class_trans_dict,sampleName,variableName)

    ## if there is not enough samples to do the external validation no matter what the user says isexternal will be false
    if len(sampleList) < 50:
        isexternal = False
    ## use hash table to see how many samples for each class and if countSample < 9 we dont do external
    hash_classCount = [0]*(classNum+1)
    for c_num in classList:
        hash_classCount[c_num] += 1
    for i in range(1,classNum+1):
        if hash_classCount[i] < 9:
            isexternal = False
    if isexternal:
        sampleList, external_validation, classList, external_class, indices_train, indices_test = selectRandom(sampleList, classList, howMuchSplit)
        # Create stat file with
        class_stat_list_noCutoff = []
        class_stat_list_external_noCutoff = []
        for classNum in range(1, int(classNum) + 1):
            class_stat_list_noCutoff.append([])
            # Create stat file with
            class_stat_list_external_noCutoff.append([])
        # Train and predict the class
        clf = svm.SVC(kernel='linear', random_state=0, probability=True)
        clf.fit(sampleList, classList)
        class_pred = clf.predict(sampleList)
        classofic_report = classification_report(classList, class_pred)
        report_lines = classofic_report.split('\n')
        report_lines = report_lines[2:]
        # generate the statistic report
        for c in range(0, classNum):
            stat_num = report_lines[c].split(' ')
            stat_num = [i for i in stat_num if i != ""]
            class_stat_list_noCutoff[c].append(stat_num[1:])
        for c in range(classNum):
            file_pkg.gen_file_by_class_matrix(["Selectivity", "Sensitivity", "Accuracy"],
                                              class_stat_list_noCutoff[c][:3], 'output/training_stat_report_class_' +
                                              [k for k, v in class_trans_dict.items() if v == str(c + 1)][
                                                  0] + '_no_cutoff.csv')
        # for external
        class_pred_external = clf.predict(external_validation)
        classofic_report_external = classification_report(external_class, class_pred_external)
        report_lines_external = classofic_report_external.split('\n')
        report_lines_external = report_lines_external[2:]
        # generate the statistic report
        for c in range(0, classNum):
            stat_num_external = report_lines_external[c].split(' ')
            stat_num_external = [i for i in stat_num_external if i != ""]
            class_stat_list_external_noCutoff[c].append(stat_num_external[1:])
        for c in range(classNum):
            file_pkg.gen_file_by_class_matrix(["Selectivity", "Sensitivity", "Accuracy"],
                                              class_stat_list_external_noCutoff[c][:3],
                                              'output/external_stat_report_class_' +
                                              [k for k, v in class_trans_dict.items() if v == str(c + 1)][
                                                  0] + '_no_cutoff.csv')
    # output the splited training and external variables in special format
    index_indices_train =  [x-1 for x in indices_train]
    index_indices_test =  [x-1 for x in indices_test]
    if isexternal:
        file_pkg.export_file(ori_sample, ori_class, index_indices_train, hori_index, 'output/training_variables.csv', class_trans_dict, sampleName,variableName)
        file_pkg.export_file(ori_sample, ori_class, index_indices_test, hori_index, 'output/external_variables.csv', class_trans_dict, sampleName,variableName)
    else:
        file_pkg.export_file(ori_sample, ori_class, index_indices_train, hori_index, 'output/training_variables.csv', class_trans_dict, sampleName,variableName)
        external_variables_wb = xlsxwriter.Workbook('output/external_variables.xlsx')
        external_variables_ws = external_variables_wb.add_worksheet()
        external_variables_ws.write(0, 0, "There is not enough samples to have external validation.")

    # get the start number and the end number
    if isCutoff:
        startNum = genStartEndNum2.cutoff_algorithm(int(classNum), classList, sampleList,V_rankingAlgorithm)
    else:
        startNum = genStartEndNum2.gaussian_algorithm(int(classNum), classList, sampleList, V_rankingAlgorithm,nComponent)

    # create a file to save the generate statistical number(accuracy, sensitivity, selectivity)
    class_stat_list = []
    for classNum in range(1, int(classNum)+1):
        class_stat_list.append([])
    class_index_list = []
    for i in range(classNum + 1):
        class_index_list.append([])
    for i in range(len(classList)):
        class_index_list[classList[i]].append(i)

    sample_taining, valid_idx = newScore.setNumber(int(classNum),classList,sampleList,startNum,howMuchSplit,V_rankingAlgorithm,nComponent)

    # generate file for selected training and selected validation in special format
    if isexternal:
        file_pkg.export_file(ori_sample, ori_class, index_indices_train, valid_idx, 'output/selected_training_variables.csv', class_trans_dict, sampleName,variableName)
        file_pkg.export_file(ori_sample, ori_class, index_indices_test, valid_idx, 'output/selected_external_variables.csv', class_trans_dict, sampleName,variableName)
    else:
        file_pkg.export_file(ori_sample, ori_class,index_indices_train,valid_idx, 'output/selected_training_variables.csv', class_trans_dict, sampleName,variableName)
        external_variables_wb = xlsxwriter.Workbook('output/selected_external_variables.xlsx')
        external_variables_ws = external_variables_wb.add_worksheet()
        external_variables_ws.write(0, 0, "There is not enough samples to have external validation.")

    # Train and predict the class
    clf = svm.SVC(kernel='linear', random_state=0, probability=True)
    clf.fit(sampleList[:,valid_idx], classList)
    class_pred = clf.predict(sampleList[:, valid_idx])
    classofic_report = classification_report(classList, class_pred)
    report_lines = classofic_report.split('\n')
    report_lines = report_lines[2:]
    # generate the statistic report
    for c in range(0, classNum):
        stat_num = report_lines[c].split(' ')
        stat_num = [i for i in stat_num if i != ""]
        class_stat_list[c].append(stat_num[1:])
    # create csv tables for the stat numbers
    for c in range(classNum):
        file_pkg.gen_file_by_class_matrix(["Selectivity", "Sensitivity", "Accuracy"],class_stat_list[c][:3],'output/training_stat_report_class_'+[k for k,v in class_trans_dict.items() if v == str(c+1)][0]+'.csv')

    ####################################  START GRAPH CODE ###################################
    # generate PCA visualization
    if scale_type == 'SVN':
        scaled_sample_training, col_mean = SVN_scale_half_data(sample_taining)
        scaled_all_sample = SVN_scale_all_data(sampleList, col_mean)
    else:
        scaled_sample_training, train_mean, train_std = scale_half_data(sample_taining)
        scaled_all_sample = scale_all_data(sampleList, train_mean, train_std)

    if scale_type == 'SVN':
        scale_training_sample, col_mean= SVN_scale_half_data(sampleList)
        scaled_external, col_mean= SVN_scale_half_data(external_validation)
    else:
        scale_training_sample, scale_training_mean, scale_training_std = scale_half_data(sampleList)
        scaled_external, scale_training_mean, scale_training_std = scale_half_data(external_validation)



    class_index_list = []
    external_class_index_list = []
    for i in range(classNum+1):
        class_index_list.append([])
        external_class_index_list.append([])
    for i in range(len(classList)):
        class_index_list[classList[i]].append(i)
    for i in range(len(external_class)):
        external_class_index_list[external_class[i]].append(i)

    class_variables = scale_training_sample[:, valid_idx]
    dummyU, dummyS, V = svds(class_variables, k=2)
    V = np.transpose(V)
    score = np.dot(scaled_all_sample[:,valid_idx], V)
    for z in range(1, classNum+1):
        class_score = score[class_index_list[z],:]
        x_ellipse, y_ellipse = confident_ellipse(class_score[:, 0], class_score[:, 1])
        plt.plot(x_ellipse, y_ellipse,color=CLASS_COLOR[z-1])
        plt.fill(x_ellipse, y_ellipse,color=CLASS_COLOR[z-1], alpha=0.3)
        class_Xt = score[class_index_list[z], :]
        plt.scatter(class_Xt[:, 0], class_Xt[:, 1], c=CLASS_COLOR[z-1], marker=CLASS_LABEL[0], label='training ' + [k for k,v in class_trans_dict.items() if v == str(z)][0])
    # calculating the PCA percentage value
    pU, pS, pV = np.linalg.svd(class_variables)
    pca_percentage_val = np.cumsum(pS) / sum(pS)
    p2_percentage = pca_percentage_val[0] * 100
    p1_percentage = pca_percentage_val[1] * 100
    plt.xlabel("PC1(%{0:0.3f}".format(p1_percentage) + ")")
    plt.ylabel("PC2 (%{0:0.3f}".format(p2_percentage) + ")")
    plt.rcParams.update({'font.size': 10})
    plt.title('PCA training')
    plt.legend()
    plt.savefig('output/pca_taining.png')
    if isexternal:
        external_Xt = np.dot(scaled_external[:,valid_idx], V)
        for n in range(1, classNum+1):
            class_external_Xt = external_Xt[external_class_index_list[n], :]
            plt.scatter(class_external_Xt[:, 0], class_external_Xt[:, 1], c=CLASS_COLOR[n-1], marker=CLASS_LABEL[1],
                               label='external ' + [k for k,v in class_trans_dict.items() if v == str(n)][0])
        clf_extern = svm.SVC(kernel='linear', random_state=0, probability=True)
        clf_extern.fit(sampleList[:,valid_idx], classList)
        class_pred = clf_extern.predict(external_validation[:, valid_idx])
        classofic_report = classification_report(external_class, class_pred)
        plt.title('PCA Training , Validation, with Feature Selection ')
        plt.rcParams.update({'font.size': 10})
        plt.legend()
        plt.savefig('output/pca_external.png')
        plt.figure().clear()
        conf_matrix = confusion_matrix(external_class, class_pred)
        file_pkg.gen_file_by_matrix(conf_matrix,'output/confusion_matrix.csv')
        clf_extern_no_FS = svm.SVC(kernel='linear', random_state=0, probability=True)
        clf_extern_no_FS.fit(sampleList, classList)
        class_pred_no_FS = clf_extern_no_FS.predict(external_validation)
        conf_matrix_no_FS = confusion_matrix(external_class, class_pred_no_FS)
        file_pkg.gen_file_by_matrix(conf_matrix_no_FS, 'output/confusion_matrix_no_FS.csv')
        report_lines = classofic_report.split('\n')
        report_lines = report_lines[2:]

        # create csv tables for the stat numbers
        for c in range(0, classNum):
            stat_num = report_lines[c].split(' ')
            stat_num = [i for i in stat_num if i != ""]
            data = stat_num[1:]
            file_pkg.gen_file_by_list_col(["Selectivity", "Sensitivity", "Accuracy"], data,
                                              'output/external_stat_report_class_' +
                                              [k for k, v in class_trans_dict.items() if v == str(c + 1)][0] + '.csv')

    # generate ROC for external validation and selected variables
    if isexternal:
        if classNum == 2:
           gen_roc_graph(sampleList[:,valid_idx],classList,external_validation[:,valid_idx],external_class,"output/rocExternal/roc_external.png", 'ROC External ')
        else:
            mul_roc_graph(classNum,class_num_label,classList,external_class,sampleList[:,valid_idx],external_validation[:,valid_idx],ROC_COLOR,"output/rocExternal/ROC External ",isMicro,'ROC External ßß', class_trans_dict)


    # generate 4 SVM graph
    # graph 1: training without feature selection
    gen_pca(scale_training_sample, classNum, class_index_list, CLASS_COLOR, CLASS_LABEL, 'output/PCATrainNoFS.png','PCA Training, No Feature Selection ',class_trans_dict)
    # generate predict ROC
    if classNum == 2:
        gen_roc_graph(sampleList,classList,sampleList,classList,'output/rocTrainNoFS/rocTrainNoFS.png','ROC Training, No Feature Selection ')
    else:
        mul_roc_graph(classNum, class_num_label, classList, classList, sampleList,
                      sampleList, ROC_COLOR, 'output/rocTrainNoFS/rocTrainNoFS',isMicro,'ROC Training, No Feature Selection ', class_trans_dict)

    # graph 2: validation without feature selection
    gen_pca(scaled_external, classNum, external_class_index_list, CLASS_COLOR, CLASS_LABEL, 'output/PCAValiNoFS.png','PCA Validation, No Feature Selection ',class_trans_dict)
    # generate predict ROC
    if classNum == 2:
        gen_roc_graph(sampleList,classList,external_validation,external_class,'output/rocValiNoFS/rocValiNoFS.png', 'ROC Validation, No Feature Selection ')
    else:
        mul_roc_graph(classNum, class_num_label, classList, external_class, sampleList,
                      external_validation, ROC_COLOR, 'output/rocValiNoFS/rocValiNoFS',isMicro,'ROC Validation, No Feature Selection ', class_trans_dict)


    # graph 3: training with feature selection
    gen_pca(scale_training_sample[:,valid_idx], classNum, class_index_list, CLASS_COLOR, CLASS_LABEL,'output/PCATrainWithFS.png','PCA Training, With Feature Selection ',class_trans_dict)
    # generate predict ROC
    if classNum == 2:
        gen_roc_graph(sampleList[:,valid_idx],classList,sampleList[:,valid_idx],classList,'output/rocTrainFS/rocTrainFS.png','ROC Training, With Feature Selection ')
    else:
        mul_roc_graph(classNum, class_num_label, classList, classList, sampleList[:,valid_idx],
                      sampleList[:, valid_idx], ROC_COLOR, 'output/rocTrainFS/rocTrainFS',isMicro, 'ROC Training, With Feature Selection ', class_trans_dict)


    # graph 4: validation with feature selection
    gen_pca(scaled_external[:, valid_idx], classNum, external_class_index_list, CLASS_COLOR, CLASS_LABEL,
            'output/PCAValiWithFS.png','PCA Validation, With Feature Selection ',class_trans_dict)

    # generate predict ROC
    if classNum == 2:
        gen_roc_graph(sampleList[:,valid_idx],classList,external_validation[:,valid_idx],external_class,'output/rocValiFS/rocValiFS.png', 'ROC Validation, With Feature Selection ' )
    else:
        mul_roc_graph(classNum, class_num_label, classList, external_class, sampleList[:, valid_idx],
                      external_validation[:, valid_idx], ROC_COLOR,'output/rocValiFS/rocValiFS',isMicro, 'ROC Validation, With Feature Selection ', class_trans_dict)
    # graph 5: PCA with Internal and external without FS
    class_variables_no_FS = scale_training_sample
    dummyU, dummyS, V_no_FS = svds(class_variables_no_FS, k=2)
    V_no_FS = np.transpose(V_no_FS)
    score_no_FS = np.dot(scaled_all_sample, V_no_FS)
    for z in range(1, classNum+1):
        class_score_no_FS = score_no_FS[class_index_list[z],:]
        x_ellipse, y_ellipse = confident_ellipse(class_score_no_FS[:, 0], class_score_no_FS[:, 1])
        plt.plot(x_ellipse, y_ellipse,color=CLASS_COLOR[z-1])
        plt.fill(x_ellipse, y_ellipse,color=CLASS_COLOR[z-1], alpha=0.3)
        class_Xt_no_FS = score_no_FS[class_index_list[z], :]
        plt.scatter(class_Xt_no_FS[:, 0], class_Xt_no_FS[:, 1], c=CLASS_COLOR[z-1], marker=CLASS_LABEL[0], label='training ' + [k for k,v in class_trans_dict.items() if v == str(z)][0])
    # calculating the PCA percentage value
    pU_no_FS, pS_no_FS, pV_no_FS = np.linalg.svd(class_variables_no_FS)
    pca_percentage_val_no_FS = np.cumsum(pS_no_FS) / sum(pS_no_FS)
    p2_percentage = pca_percentage_val_no_FS[0] * 100
    p1_percentage = pca_percentage_val_no_FS[1] * 100
    plt.xlabel("PC1(%{0:0.3f}".format(p1_percentage) + ")")
    plt.ylabel("PC2 (%{0:0.3f}".format(p2_percentage) + ")")
    plt.rcParams.update({'font.size': 10})
    if isexternal:
        external_Xt = np.dot(scaled_external, V_no_FS)
        for n in range(1, classNum+1):
            class_external_Xt = external_Xt[external_class_index_list[n], :]
            plt.scatter(class_external_Xt[:, 0], class_external_Xt[:, 1], c=CLASS_COLOR[n-1], marker=CLASS_LABEL[1],
                               label='external ' + [k for k,v in class_trans_dict.items() if v == str(n)][0])
    plt.title('PCA Training , Validation, No Feature Selection')
    plt.rcParams.update({'font.size': 10})
    plt.legend()
    plt.savefig('output/pca_external_No_FS.png',bbox_inches="tight" )
    plt.figure().clear()

    ####################################  END GRAPH CODE ###################################
## axis=0 is for row
## axis=1 is for column
def scale_half_data(samples):
    # after get all the selected variables we make them a metrix and calculate the mean
    samples = np.array(samples)
    samples_mean = samples.mean(axis=0)
    samples_std = np.std(samples, axis=0)
    np.set_printoptions(threshold=sys.maxsize)
    functionTop = np.subtract(samples, samples_mean)
    scaled_samples = np.divide(functionTop, samples_std)
    scaled_samples = np.nan_to_num(scaled_samples, nan=(10**-12))
    for list in scaled_samples:
        list[list==inf] = 10**-12
    return scaled_samples, samples_mean, samples_std

def scale_all_data(samples,mean,std):
    functionTop = np.subtract(samples, mean)
    scaled_samples = np.divide(functionTop, std)
    for list in scaled_samples:
        list[list==inf] = 10**-12
    scaled_samples = np.nan_to_num(scaled_samples, nan=(10**-12))
    return scaled_samples

def SVN_scale_half_data(samples):
    sd = StandardScaler(with_mean=True, with_std=False)
    sd.fit(samples)
    col_mean = sd.mean_
    scaled_samples = scale(samples, axis=1, with_mean=True, with_std=True)
    return scaled_samples, col_mean

def SVN_scale_all_data(samples,col_mean):
    scaled_samples = scale(samples, axis=1, with_mean=True, with_std=True)
    scaled_samples = np.subtract(scaled_samples, col_mean)

    return scaled_samples

def selectRandom(sample_list,class_list,howMuchSplit):
    indices = np.arange(1,len(class_list)+1)
    sample_matrix = np.array(sample_list)
    class_matrix = np.array(class_list)
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(sample_matrix, class_matrix, indices, test_size=float(howMuchSplit), stratify=class_matrix)
    return X_train, X_test, y_train, y_test, indices_train, indices_test

def confident_ellipse(score1, score2, confident_interval = 0.95):
    score1 = np.array(score1)
    score2 = np.array(score2)
    chi_2 = chi2.ppf(confident_interval, df=2)
    d1 = score1.mean(axis=0)
    d2 = score2.mean(axis=0)
    data = [score1,score2]
    covMat = np.cov(data)
    eivec, eigval, Vh1 = np.linalg.svd(covMat)
    phi1 = math.atan2(eivec[0][1], eivec[0][0])
    if phi1 < 0:
        phi1 = phi1 + 2*math.pi
    theta = np.arange(0, 2*math.pi, 0.01)
    x_ellipse = []
    y_ellipse = []

    for i in theta:
        x_temp = d1 + math.sqrt(chi_2) * math.sqrt(eigval[0]) * math.cos(i) * math.cos(phi1) - math.sqrt(chi_2) * math.sqrt(eigval[1]) * math.sin(i) * math.sin(phi1)
        y_temp = d2 + math.sqrt(chi_2) * math.sqrt(eigval[0]) * math.cos(i) * math.sin(phi1) + math.sqrt(chi_2) * math.sqrt(eigval[1]) * math.sin(i) * math.cos(phi1)
        y_ellipse.append(y_temp)
        x_ellipse.append(x_temp)
    return x_ellipse, y_ellipse
def mul_roc_graph(classNum, class_num_label, trainingClass, predicClass, trainingVal, predicVal, roc_colors, output_filename,isMicro,graph_title,class_trans_dict):
    figPlots = []
    for w in range(classNum):
        figPlots.append(plt.subplots(1))
    training_class = label_binarize(trainingClass, classes=class_num_label)
    predict_class = label_binarize(predicClass, classes=class_num_label)
    classifier = OneVsRestClassifier(
        svm.SVC(kernel="linear", probability=True, random_state=0)
    )
    y_score = classifier.fit(trainingVal, training_class).decision_function(
        predicVal)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classNum):
        fpr[i], tpr[i], _ = metrics.roc_curve(predict_class[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(predict_class.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    if isMicro:
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            color=str(roc_colors[1]),
        )
    else:
        for k in range(classNum):
            figPlots[k][1].plot(
                fpr[k],
                tpr[k],
                color=str(roc_colors[1]),
                label="ROC curve (area = %0.3f)" % roc_auc[k],
            )
            figPlots[k][1].legend()
    plt.rcParams.update({'font.size': 14})

    for j in range(classNum):
        figPlots[j][1].set_title(graph_title +'class '+ [k for k,v in class_trans_dict.items() if v == str(j+1)][0])
        figPlots[j][0].savefig(output_filename +'class '+ [k for k,v in class_trans_dict.items() if v == str(j+1)][0] + '.png')
        figPlots[j][0].clear()


def gen_roc_graph(training_sample,training_class,predict_sample,predict_class, fileName, graph_title):
    # Create a svm Classifier
    clf = svm.SVC(kernel='linear', random_state=0, probability=True)  # Linear Kernel
    # Train the model using the training sets
    clf.fit(training_sample, training_class)
    class_pred = clf.predict_proba(predict_sample)
    class_pred = class_pred[:, 1]
    auc_external = metrics.roc_auc_score(predict_class, class_pred)
    fpr, tpr, _ = metrics.roc_curve(predict_class, class_pred, pos_label=2)
    plt.plot(fpr, tpr, label="micro-average ROC curve (area = {0:0.3f})".format(auc_external))
    plt.title(graph_title)
    plt.rcParams.update({'font.size': 14})
    plt.legend(loc=4)
    plt.savefig(fileName,bbox_inches="tight")
    plt.figure().clear()

def gen_pca(training_sample,classNum,class_index_list,class_color,class_label,fileName,graph_title, class_trans_dict):
    dummyU, dummyS, V = svds(training_sample, k=2)
    V = np.transpose(V)
    Xt_training_noFS = np.dot(training_sample, V)
    for z in range(1, classNum + 1):
        class_Xt_training_noFS = Xt_training_noFS[class_index_list[z], :]
        x_ellipse, y_ellipse = confident_ellipse(class_Xt_training_noFS[:, 0], class_Xt_training_noFS[:, 1])
        plt.plot(x_ellipse, y_ellipse, color=class_color[z - 1])
        plt.fill(x_ellipse, y_ellipse, color=class_color[z - 1], alpha=0.3)
        plt.scatter(class_Xt_training_noFS[:, 0], class_Xt_training_noFS[:, 1], c=class_color[z - 1],
                    marker=class_label[0], label='class ' + [k for k,v in class_trans_dict.items() if v == str(z)][0])
    # calculating the PCA percentage value
    pU, pS, pV = np.linalg.svd(training_sample)
    pca_percentage_val = np.cumsum(pS) / sum(pS)
    p2_percentage = pca_percentage_val[0] * 100
    p1_percentage = pca_percentage_val[1] * 100
    plt.xlabel("PC1(%{0:0.3f}".format(p1_percentage) + ")")
    plt.ylabel("PC2 (%{0:0.3f}".format(p2_percentage) + ")")
    plt.title(graph_title)
    plt.rcParams.update({'font.size': 10})
    plt.legend()
    plt.savefig(fileName,bbox_inches="tight")
    plt.figure().clear()

def class_tupa(X,Y):
    cls = np.array(Y)
    X = np.array(X)
    cls = np.unique(cls)
    numCls = len(cls)
    class_idx = []
    return_sampleList = []
    for j in range(numCls):
        class_idx.append([])
    for z in range(len(Y)):
        class_idx[Y[z]-1].append(z)
    usetupa =[]
    for i in range(numCls):
        temp_usetupa =[]
        X_temp = X[class_idx[i],:]
        for i in range(len(X_temp[0])):
            temp_vari = X_temp[:,i]
            if 0 not in temp_vari:
                temp_usetupa.append(1)
            else:
                temp_usetupa.append(0)
        usetupa.append(temp_usetupa)
    for i in range(len(Y)):
        X_temp = X[i]
        mul_temp = copy.copy(X_temp)
        for k in range(len(mul_temp)):
            mul_temp[k] = mul_temp[k]*usetupa[Y[i]-1][k]
        temp_sum = np.sum(mul_temp)
        temp_div = np.divide(X_temp,temp_sum)
        return_sampleList.append(temp_div.tolist())
    return return_sampleList

def tupa(X,Y):
    cls = np.array(Y)
    X = np.array(X)
    cls = np.unique(cls)
    usetupa =[]
    return_sampleList=[]
    for i in range(len(X[0])):
        temp_vari = X[:,i]
        if 0 not in temp_vari:
            usetupa.append(1)
        else:
            usetupa.append(0)
    for i in range(len(Y)):
        X_temp = X[i]
        mul_temp = copy.copy(X_temp)
        for k in range(len(mul_temp)):
            mul_temp[k] = mul_temp[k]*usetupa[k]
        temp_sum = np.sum(mul_temp)
        temp_div = np.divide(X_temp,temp_sum)
        return_sampleList.append(temp_div.tolist())
    return return_sampleList

# isexternal,howMuchSplit,isMicro,tupaType,isMotabo,MotaboFileName,DataFileName,ClassFileName,sampleNameFile,variableNameFile
## Tupa Selection: tupa, classtupa, notupa
## Scale Selection: SNV,AutoScale
#main(True,0.5,False,'notupa',False,'Input/mota_data.csv','Input/data_pureOil.csv','Input/class_pureOil_2.csv','Input/sampleName_pureOil.csv','Input/Vname_pureOil.csv','AutoScale',10,0.85)
#main(True,0.5,False,'classTupa',False,'Input/mota_data.csv','Input/data_algae.csv','Input/class_algae_string.csv','Input/S_name.csv','Input/v_name.csv','AutoScale',10,0.85)

## Needed parameters
## 1 is external
## 2 the rate of splite the trainning and validation
## 3 is Micro for ROC
## 4 What kind of TUPA : a) classtupa b)tupa c)notupa
## 5 is the input data from motabo analize
## 6 your motabo data file name, if not using motabo data just input None instead
## 7 your data file name (not motabo data )
## 8 your class file name (not motabo data )
## 9 your sample name file name (not motabo data )
## 10 your variable name file name (not motabo data )
## 11 how would you like to scale your data: a) AotuScale b)SVN
## 12 how many iterations you like
## 13 the survival rate
if __name__ == "__main__":
    if sys.argv[1].lower() == 'true':
        isexternal = True
    else:
        isexternal = False
    rateSplit = float(sys.argv[2])
    if sys.argv[3].lower() == 'true':
        isMicro = True
    else:
        isMicro = False
    tupaType = sys.argv[4]
    if sys.argv[5].lower() == 'true':
        isMotabo = True
    else:
        isMotabo = False
    motaboFileName = sys.argv[6]
    dataFileName = sys.argv[7]
    classFileName = sys.argv[8]
    sampleNameFileName = sys.argv[9]
    variableNameFileName = sys.argv[10]
    scaleType = sys.argv[11]
    howManyIteration = int(sys.argv[12])
    survivalrate = float(sys.argv[13])
    V_rankingAlgorithm = sys.argv[14]
    nComponent = int(sys.argv[15])
    if sys.argv[16].lower() == 'true':
        isCutoff =  True
    else:
        isCutoff =  False


    main(isexternal,rateSplit,isMicro,tupaType,isMotabo,motaboFileName,dataFileName,classFileName,sampleNameFileName,variableNameFileName,scaleType,howManyIteration,survivalrate,V_rankingAlgorithm,nComponent,isCutoff)