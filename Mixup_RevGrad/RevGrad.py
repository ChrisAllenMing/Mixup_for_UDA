from __future__ import print_function
import os, sys
import math
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import model_zoo

import data_loader
import models

###################################

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', required = True, help = 'root to the data')
parser.add_argument('--source', type = str, default = 'usps', help = 'the source domain')
parser.add_argument('--target', type = str, default = 'mnist', help = 'the target domain')
parser.add_argument('--model_dir', type = str, default = './models/', help = 'the path to save models')
parser.add_argument('--batch_size', type = int, default = 100, help = 'the size of mini-batch')
parser.add_argument('--epochs', type = int, default = 100, help = 'the number of epochs')
parser.add_argument('--lr', type = float, default = 0.001, help = 'the initial learning rate')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'the momentum of gradient')
parser.add_argument('--l2_decay', type = float, default = 5e-4, help = 'the l2 decay used in training')
parser.add_argument('--seed', type = int, default = 100, help = 'the manual seed')
parser.add_argument('--log_interval', type = int, default = 50, help = 'the interval of print')
parser.add_argument('--gpu_id', type = str, default = '0', help = 'the gpu device id')

opt = parser.parse_args()
print (opt)

###################################

# Training settings
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed(opt.seed)

# Dataloader

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source_loader = data_loader.load_training(opt.root_path, opt.source, opt.batch_size, kwargs)
target_train_loader = data_loader.load_training(opt.root_path, opt.target, opt.batch_size, kwargs)
target_test_loader = data_loader.load_testing(opt.root_path, opt.target, opt.batch_size, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)
nclasses = len(source_loader.dataset.classes)

###################################

# For every epoch training
def train(epoch, model):

    optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    data_source_iter = iter(source_loader)
    data_target_iter = iter(target_train_loader)
    dlabel_src = Variable(torch.ones(opt.batch_size).long().cuda())
    dlabel_tgt = Variable(torch.zeros(opt.batch_size).long().cuda())

    i = 1
    while i <= len_source_loader:
        model.train()

        # the parameter for reversing gradients
        p = float(i + epoch * len_source_loader) / opt.epochs / len_source_loader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # for the source domain batch
        source_data, source_label = data_source_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)

        _, clabel_src, dlabel_pred_src = model(source_data, alpha = alpha)
        label_loss = loss_class(clabel_src, source_label)
        domain_loss_src = loss_domain(dlabel_pred_src, dlabel_src)

        # for the target domain batch
        target_data, target_label = data_target_iter.next()
        if i % len_target_loader == 0:
            data_target_iter = iter(target_train_loader)
        if cuda:
            target_data, target_label = target_data.cuda(), target_label.cuda()
        target_data = Variable(target_data)

        _, clabel_tgt, dlabel_pred_tgt = model(target_data, alpha = alpha)
        domain_loss_tgt = loss_domain(dlabel_pred_tgt, dlabel_tgt)

        domain_loss_total = domain_loss_src + domain_loss_tgt
        loss_total = label_loss + domain_loss_total

        optimizer.zero_grad()
        # label_loss.backward()
        loss_total.backward()
        optimizer.step()

        if i % opt.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlabel_Loss: {:.6f}\tdomain_Loss: {:.6f}'.format(
                epoch, i * len(source_data), len_source_dataset,
                100. * i / len_source_loader, label_loss.item(), domain_loss_total.item()))
        i = i + 1


# For every epoch evaluation
def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in target_test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        _, s_output, t_output = model(data, alpha = 0)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target, size_average=False).item()
        pred = s_output.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len_target_dataset

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        opt.target, test_loss, correct, len_target_dataset,
        100. * correct / len_target_dataset))

    return correct


if __name__ == '__main__':

    model = models.RevGrad(num_classes = nclasses)
    print (model)

    max_correct = 0
    if cuda:
        model.cuda()

    # start training
    for epoch in range(1, opt.epochs + 1):
        train(epoch, model)
        # test for every epoch
        t_correct = test(model)
        if t_correct > max_correct:
            max_correct = t_correct
            if not os.path.exists(opt.model_dir):
                os.mkdir(opt.model_dir)
            torch.save(model.state_dict(), os.path.join(opt.model_dir, 'best_model.pkl'))

        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
              opt.source, opt.target, max_correct, 100. * max_correct / len_target_dataset ))