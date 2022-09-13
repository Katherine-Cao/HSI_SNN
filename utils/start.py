# -*- coding: utf-8 -*-
"""
@Author: Pangpd (https://github.com/pangpd/DS-pResNet-HSI)
@UsedBy: Katherine_Cao (https://github.com/Katherine-Cao/HSI_SNN)
"""

import numpy as np
from utils import evaluate
import torch
import torch.nn.parallel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    accs = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        model.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)  # CrossEntropyloss

        losses[batch_idx] = loss.item()
        accs[batch_idx] = evaluate.accuracy(outputs.data, targets.data)[0].item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (np.average(losses), np.average(accs))



def test(testloader, model, criterion, epoch, use_cuda):
    model.eval()
    accs = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs = model(inputs)
            losses[batch_idx] = criterion(outputs, targets).item()     # CrossEntropyLoss
            accs[batch_idx] = evaluate.accuracy(outputs.data, targets.data, topk=(1,))[0].item()
    return (np.average(losses), np.average(accs))


def predict(test_loader, model, use_cuda):
    model.eval()
    predicted = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda: inputs = inputs.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            [predicted.append(a) for a in model(inputs).data.cpu().numpy()]
    return np.array(predicted)


def adjust_learning_rate(optimizer, epoch, learn_rate):
    lr = learn_rate * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))  # 1-149:0.1，150-200:0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
