# -*- coding: utf-8 -*-
"""
@Author: Pangpd (https://github.com/pangpd/DS-pResNet-HSI)
@UsedBy: Katherine_Cao (https://github.com/Katherine-Cao/HSI_SNN)
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import matplotlib.patches as mpatches


def color_list(class_nums):
    """
    :param class_nums: 地物类数
    :return: 返回实际的地物数量的颜色版
    """
    colors = np.array([[0, 0, 0], [254, 0, 0], [0, 255, 1], [2, 2, 255],
                       [255, 255, 2], [2, 255, 255], [255, 2, 255], [193, 193, 193],
                       [129, 129, 129], [129, 2, 2], [128, 129, 2], [2, 129, 3],
                       [130, 2, 128], [2, 129, 131], [3, 2, 129], [255, 166, 2],
                       [255, 216, 2]])
    return colors[:class_nums + 1]

def show_label(data, labels, class_nums, save_path, class_names=None):
    '''
    只画出标签图
    :param data: loadData函数中返回的data（经过标准化后的）
    :param labels: 地物标签值
    :param class_nums: 地物总类数
    :param save_path: gt图的保存路径
    :param class_names: 地物标签名
    :return: None
    '''
    number_of_rows = int(data.shape[0])
    number_of_columns = int(data.shape[1])
    gt = labels.reshape(-1, 1)  # 拉成一列
    colors = color_list(class_nums)
    colors = sklearn.preprocessing.minmax_scale(colors, feature_range=(0, 1))
    gt_thematic_map = np.zeros(shape=(number_of_rows, number_of_columns, 3))
    cont = 0
    for i in range(number_of_rows):
        for j in range(number_of_columns):
            gt_thematic_map[i, j, :] = colors[gt[cont, 0]]
            cont += 1
    plt.axis('off')
    height, width = labels.shape

    plt.gcf().set_size_inches(width / 100.0, height / 100.0)  # 输出width*height像素
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=0.999, left=0, hspace=0, wspace=0)
    #plt.margins(0, 0)
    plt.xticks([])  # 关闭刻度
    plt.yticks([])
    plt.imshow(gt_thematic_map)  # 标签图
    plt.savefig(save_path, dpi=300)

def show_gt(data, labels, class_nums, save_path, class_names=None, only_color=False):
    '''
    画出标签图的RGB图和假彩色图(任意非R,G,B波段的合成图，)
    :param data: loadData函数中返回的data（经过标准化后的）
    :param labels: 地物标签值
    :param class_nums: 地物总类数
    :param save_path: gt图的保存路径
    :param class_names: 地物标签名
    :return: None
    '''
    number_of_rows = int(data.shape[0])
    number_of_columns = int(data.shape[1])
    data = data.reshape(-1, data.shape[-1])
    gt = labels.reshape(-1, 1)  # 拉成一列
    colors = color_list(class_nums)
    # visualizing the indian pines dataset hyperspectral cube in RGB

    colors = sklearn.preprocessing.minmax_scale(colors, feature_range=(0, 1))
    pixels_normalized = sklearn.preprocessing.minmax_scale(data, feature_range=(0, 1))

    gt_thematic_map = np.zeros(shape=(number_of_rows, number_of_columns, 3))
    rgb_hyperspectral_image = np.zeros(shape=(number_of_rows, number_of_columns, 3))
    cont = 0
    for i in range(number_of_rows):
        for j in range(number_of_columns):
            rgb_hyperspectral_image[i, j, 0] = pixels_normalized[cont, 29]  # 10
            rgb_hyperspectral_image[i, j, 1] = pixels_normalized[cont, 42]  # 24
            rgb_hyperspectral_image[i, j, 2] = pixels_normalized[cont, 89]  # 44
            gt_thematic_map[i, j, :] = colors[gt[cont, 0]]
            cont += 1
    # fig = plt.figure(figsize=(15, 15))
    if only_color == True:  # 仅获取假彩色图像

        plt.axis('off')
        height, width =labels.shape

        plt.gcf().set_size_inches(width / 100.0, height / 100.0)  # 输出width*height像素
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=0.999, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.xticks([])  # 关闭刻度
        plt.yticks([])

        plt.imshow(rgb_hyperspectral_image)  # 假彩色
        plt.savefig(save_path, dpi=300)
        return

    fig = plt.figure()
    columns = 2
    rows = 1

    ax1 = fig.add_subplot(rows, columns, 1)
    plt.xticks([])  # 关闭刻度
    plt.yticks([])
    ax1.set_xlabel("(a)", fontsize=15)
    plt.imshow(rgb_hyperspectral_image)  # 假彩色

    ax2 = fig.add_subplot(rows, columns, 2)
    fig.subplots_adjust(left=0.01, top=0.96, right=0.96, bottom=0.04, wspace=0.02, hspace=0.04)
    plt.xticks([])  # 关闭刻度
    plt.yticks([])
    ax2.set_xlabel("(b)", fontsize=15)
    plt.imshow(gt_thematic_map)  # 标签图

    patches = [mpatches.Patch(color=colors[i + 1], label=class_names[i]) for i in range(len(colors) - 1)]
    # plt.legend(handles=patches, loc=4, borderaxespad=0.)
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0, frameon=False)
    # plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def show_pred(pred_test, labels, class_nums, dataset_name, save_path, removeZeroLabels=True):
    '''
    画出预测后的结果图，带背景和不带背景,默认使用不带背景
    :param pred_test: 预测后的结果（是行元素）
    :param labels: 真实标签值
    :param class_nums: 地物种类数
    :param dataset_name: 数据集名字
    :param save_path: 保存路径 数据集名字
    :param removeZeroLabels: 是否移除背景，True表示显示不带背景图，False表示显示背景图
    :return:  None
    '''
    pred_map = np.zeros(shape=(labels.shape[0], labels.shape[1], 3))
    x = np.ravel(pred_test).reshape(-1, 1)  # 预测的结果也拉成一列
    colors = color_list(class_nums)  # 获取染色板
    colors = sklearn.preprocessing.minmax_scale(colors, feature_range=(0, 1))
    cont = 0
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            pred_map[i, j, :] = colors[x[cont, 0]]  # 染色
            cont += 1
    fig = plt.figure(figsize=(15, 15))

    height, width = labels.shape
    plt.axis('off')
    plt.gcf().set_size_inches(width / 100.0, height / 100.0)  # 输出width*height像素
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=0.999, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.xticks([])  # 关闭刻度
    plt.yticks([])
    plt.imshow(pred_map)  # 带背景的预测结果图
    fig.savefig(save_path + '/maps_withZeros' + '_' + dataset_name + '.png', dpi=300)

    if removeZeroLabels == True:  # 显示不带背景的预测结果图
        pred_test_2D = pred_test.reshape(labels.shape[0], labels.shape[1])  # 转换成2维
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] == 0:
                    pred_test_2D[i, j] = 0
        pred_test_1D = pred_test_2D.reshape(-1, 1)  # 再次将结果拉成一列
        pred_map_removeZeroLabels = np.zeros(shape=(labels.shape[0], labels.shape[1], 3))
        cont = 0
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                pred_map_removeZeroLabels[i, j, :] = colors[pred_test_1D[cont, 0]]  # 染色
                cont += 1
        fig = plt.figure(figsize=(15, 15))

        plt.axis('off')
        plt.gcf().set_size_inches(width / 100.0, height / 100.0)  # 输出width*height像素
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=0.999, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.xticks([])  # 关闭刻度
        plt.yticks([])
        plt.imshow(pred_map_removeZeroLabels)  # 不带背景的预测结果图
        fig.savefig(save_path + '/maps_pred' + '_' + dataset_name + '.png', dpi=300)
