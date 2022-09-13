# -*- coding: utf-8 -*-
"""
@Author: Pangpd (https://github.com/pangpd/DS-pResNet-HSI)
@UsedBy: Katherine_Cao (https://github.com/Katherine-Cao/HSI_SNN)
"""


import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(y_pred, y_test):
    classification = classification_report(y_test, y_pred)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    return classification, confusion, oa * 100, aa * 100, kappa * 100, list(x * 100 for x in list(each_acc))


def stats(oa_list, aa_list, kappa_list, each_acc_list, train_time_list, test_time_list):
    '''

    :param oa_list:  多次实验的OA构成的list
    :param aa_list:   多次实验AA构成的list
    :param kappa_list: 多次实验kappa构成的list
    :param each_acc_list: 多次每类地物分类精度构成的多维数组
    :param train_time_list: 多次实验训练时间构成的list
    :param test_time_list: 多次实验测试时间构成的list
    :return: OA,AA，kappa，每类地物分类精度,训练时间，测试时间，总体标准差，样本标准差
    '''
    each_acc_list = np.array(each_acc_list)  # 转为np数组

    stats_oa = {}
    stats_aa = {}
    stats_kappa = {}
    stats_each_acc = {}
    stats_train_time = {}
    stats_test_time = {}

    average_oa = np.around(np.mean(oa_list), decimals=2)  # oa
    overall_std_oa = np.around(np.std(oa_list), decimals=2)  # 总体标准差（/n）
    samp_std_oa = np.around(np.std(oa_list, ddof=1), decimals=2)  # 样本标准差(/n-1)
    stats_oa['av_oa'] = average_oa
    stats_oa['ov_std_oa'] = overall_std_oa
    stats_oa['samp_std_oa'] = samp_std_oa

    average_aa = np.around(np.mean(aa_list), decimals=2)  # aa
    overall_std_aa = np.around(np.std(aa_list), decimals=2)
    samp_std_aa = np.around(np.std(aa_list, ddof=1), decimals=2)
    stats_aa['av_aa'] = average_aa
    stats_aa['ov_std_aa'] = overall_std_aa
    stats_aa['samp_std_aa'] = samp_std_aa

    average_kappa = np.around(np.mean(kappa_list), decimals=2)  # kappa
    overall_std_kappa = np.around(np.std(kappa_list), decimals=2)
    samp_std_kappa = np.around(np.std(kappa_list, ddof=1), decimals=2)
    stats_kappa['av_kappa'] = average_kappa
    stats_kappa['ov_std_kappa'] = overall_std_kappa
    stats_kappa['samp_std_kappa'] = samp_std_kappa

    average_each_acc_list = np.around(each_acc_list.mean(axis=0), decimals=2)  # each_acc
    overall_std_each_acc_list = np.round(each_acc_list.std(axis=0), decimals=2)
    samp_std_each_acc_list = np.round(each_acc_list.std(axis=0, ddof=1), decimals=2)
    stats_each_acc['av_each_acc'] = average_each_acc_list
    stats_each_acc['ov_std_each_acc'] = overall_std_each_acc_list
    stats_each_acc['samp_std_each_acc'] = samp_std_each_acc_list

    average_train_time = np.around(np.mean(train_time_list), decimals=2)  # train_time
    overall_std_train_time = np.around(np.std(train_time_list), decimals=2)
    samp_std_train_time = np.around(np.std(train_time_list, ddof=1), decimals=2)
    stats_train_time['av_train_time'] = average_train_time
    stats_train_time['ov_std_train_time'] = overall_std_train_time
    stats_train_time['samp_std_train_time'] = samp_std_train_time

    average_test_time = np.around(np.mean(test_time_list), decimals=2)  # test_time
    overall_std_test_time = np.around(np.std(test_time_list), decimals=2)
    samp_std_test_time = np.around(np.std(test_time_list, ddof=1), decimals=2)
    stats_test_time['av_test_time'] = average_test_time
    stats_test_time['ov_std_test_time'] = overall_std_test_time
    stats_test_time['samp_std_test_time'] = samp_std_test_time

    return stats_oa, stats_aa, stats_kappa, stats_each_acc, stats_train_time, stats_test_time
