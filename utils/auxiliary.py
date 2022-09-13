# -*- coding: utf-8 -*-
"""
@Author: Pangpd (https://github.com/pangpd/DS-pResNet-HSI)
@UsedBy: Katherine_Cao (https://github.com/Katherine-Cao/HSI_SNN)
"""


import torch
from graphviz import Digraph
from torch.autograd import Variable
import matplotlib.pyplot as plt


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


import logging


def get_logger(logger_name, log_dir):
    """获取日志对象"""
    log_format = '[%(asctime)s] %(message)s'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_dir + '/' + 'result_' + logger_name + '.txt', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        # StreamHandler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter(log_format))
        # logger绑定处理对象
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def save_acc_loss(train_acc_list, train_loss_list, valid_acc_list, valid_loss_list, save_path):
    # 显示训练和测试的结果图
    iters = range(0, len(valid_acc_list))
    plt.figure()
    plt.plot(iters, list(map(lambda x: x / 100, train_acc_list)), 'r', label='train acc')
    plt.plot(iters, train_loss_list, 'g', label='train loss')
    plt.plot(iters, list(map(lambda x: x / 100, valid_acc_list)), 'b', label='val acc')
    plt.plot(iters, valid_loss_list, 'k', label='val loss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc/loss')
    plt.legend(loc="upper right")
    plt.savefig(save_path + '/acc_loss.png')
   # plt.show()

def save_acc_loss2(train_acc_list, train_loss_list, valid_acc_list, valid_loss_list, save_path):
    # 显示训练和测试的结果图
    iters = range(0, len(valid_acc_list))
    plt.figure()
    plt.plot(iters, list(map(lambda x: x / 1, train_acc_list)), 'r', label='train acc')
    plt.plot(iters, list(map(lambda x: x / 1, valid_acc_list)), 'b', label='val acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.savefig(save_path + '/acc_1.png')

    plt.clf()
    plt.plot(iters, valid_loss_list, 'k', label='val loss')
    plt.plot(iters, train_loss_list, 'g', label='train loss')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.savefig(save_path + '/loss_1.png')
