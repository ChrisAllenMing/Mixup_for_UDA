import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import torch.optim as optim
import torchvision.utils as vutils
import itertools, datetime
import numpy as np
import models
import utils
import random
import os, sys


class DM_ADA(object):

    def __init__(self, opt, nclasses, mean, std, source_trainloader, source_valloader, target_trainloader,
                 target_valloader):

        self.source_trainloader = source_trainloader
        self.source_valloader = source_valloader
        self.target_trainloader = target_trainloader
        self.target_valloader = target_valloader
        self.opt = opt
        self.mean = mean
        self.std = std
        self.best_val = 0

        # Defining networks and optimizers
        self.nclasses = nclasses
        self.netG = models._netG(opt, nclasses)
        self.netD = models._netD(opt, nclasses)
        self.netF = models._netF(opt)
        self.netC = models._netC(opt, nclasses)

        # Weight initialization
        self.netG.apply(utils.weights_init)
        self.netD.apply(utils.weights_init)
        self.netF.apply(utils.weights_init)
        self.netC.apply(utils.weights_init)

        # Defining loss criterions
        self.criterion_c = nn.CrossEntropyLoss()
        self.criterion_s = nn.BCELoss()

        if opt.gpu >= 0:
            self.netD.cuda()
            self.netG.cuda()
            self.netF.cuda()
            self.netC.cuda()
            self.criterion_c.cuda()
            self.criterion_s.cuda()

        # Defining optimizers
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(0.8, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.8, 0.999))
        self.optimizerF = optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(0.8, 0.999))
        self.optimizerC = optim.Adam(self.netC.parameters(), lr=opt.lr, betas=(0.8, 0.999))

        # Other variables
        self.real_label_val = 1
        self.fake_label_val = 0

    """
    Validation function
    """

    def validate(self, epoch):

        self.netF.eval()
        self.netC.eval()
        total = 0
        correct = 0

        # Testing the model
        for i, datas in enumerate(self.target_valloader):
            inputs, labels = datas
            inputv, labelv = Variable(inputs.cuda()), Variable(labels.cuda())

            embedding, mean, std = self.netF(inputv)
            mean_std = torch.cat((mean, std), 1)
            outC_logit, _ = self.netC(mean_std)
            _, predicted = torch.max(outC_logit.data, 1)
            total += labels.size(0)
            correct += ((predicted == labels.cuda()).sum())

        val_acc = 100 * float(correct) / total

        # Saving checkpoints
        if val_acc > self.best_val:
            self.best_val = val_acc
            torch.save(self.netF.state_dict(), '%s/models/model_best_netF.pth' % (self.opt.outf))
            torch.save(self.netC.state_dict(), '%s/models/model_best_netC.pth' % (self.opt.outf))
            torch.save(self.netD.state_dict(), '%s/models/model_best_netD.pth' % (self.opt.outf))
            torch.save(self.netG.state_dict(), '%s/models/model_best_netG.pth' % (self.opt.outf))

        # Print the validation information
        print('%s| Epoch: %d, Correct/Total: %d / %d, Val Accuracy: %f, Best Accuracy: %f %%\n' \
              % (datetime.datetime.now(), epoch, correct, total, val_acc, self.best_val))

    """
    Train function
    """

    def train(self):

        curr_iter = 0

        reallabel = torch.FloatTensor(self.opt.batchSize).fill_(self.real_label_val)
        fakelabel = torch.FloatTensor(self.opt.batchSize).fill_(self.fake_label_val)
        if self.opt.gpu >= 0:
            reallabel = reallabel.cuda()
            fakelabel = fakelabel.cuda()
        reallabelv = Variable(reallabel)
        fakelabelv = Variable(fakelabel)

        for epoch in range(self.opt.nepochs):

            self.netG.train()
            self.netF.train()
            self.netC.train()
            self.netD.train()

            for i, (datas, datat) in enumerate(itertools.izip(self.source_trainloader, self.target_trainloader)):

                ###########################
                # Forming input variables
                ###########################

                src_inputs, src_labels = datas
                tgt_inputs, __ = datat
                src_inputs_unnorm = (((src_inputs * self.std[0]) + self.mean[0]) - 0.5) * 2
                tgt_inputs_unnorm = (((tgt_inputs * self.std[0]) + self.mean[0]) - 0.5) * 2

                # Creating one hot vector
                labels_onehot = np.zeros((self.opt.batchSize, self.nclasses + 1), dtype=np.float32)
                for num in range(self.opt.batchSize):
                    labels_onehot[num, src_labels[num]] = 1
                src_labels_onehot = torch.from_numpy(labels_onehot)

                labels_onehot = np.zeros((self.opt.batchSize, self.nclasses + 1), dtype=np.float32)
                for num in range(self.opt.batchSize):
                    labels_onehot[num, self.nclasses] = 1
                tgt_labels_onehot = torch.from_numpy(labels_onehot)

                # feed variables to gpu
                if self.opt.gpu >= 0:
                    src_inputs, src_labels = src_inputs.cuda(), src_labels.cuda()
                    src_inputs_unnorm = src_inputs_unnorm.cuda()
                    tgt_inputs_unnorm = tgt_inputs_unnorm.cuda()
                    tgt_inputs = tgt_inputs.cuda()
                    src_labels_onehot = src_labels_onehot.cuda()
                    tgt_labels_onehot = tgt_labels_onehot.cuda()

                # Wrapping in variable
                src_inputsv, src_labelsv = Variable(src_inputs), Variable(src_labels)
                src_inputs_unnormv = Variable(src_inputs_unnorm)
                tgt_inputsv = Variable(tgt_inputs)
                tgt_inputs_unnormv = Variable(tgt_inputs_unnorm)
                src_labels_onehotv = Variable(src_labels_onehot)
                tgt_labels_onehotv = Variable(tgt_labels_onehot)

                ###########################
                # Updates
                ###########################

                # Mix source and target domain images
                mix_ratio = np.random.beta(self.opt.alpha, self.opt.alpha)
                mix_ratio = round(mix_ratio, 2)
                # clip the mixup_ratio
                if (mix_ratio >= 0.5 and mix_ratio < (0.5 + self.opt.clip_thr)):
                    mix_ratio = 0.5 + self.opt.clip_thr
                if (mix_ratio > (0.5 - self.opt.clip_thr) and mix_ratio < 0.5):
                    mix_ratio = 0.5 - self.opt.clip_thr

                # Define labels for mixed images
                mix_label = torch.FloatTensor(self.opt.batchSize).fill_(mix_ratio)
                if self.opt.gpu >= 0:
                    mix_label = mix_label.cuda()
                mix_labelv = Variable(mix_label)

                mix_samples = mix_ratio * src_inputs_unnormv + (1 - mix_ratio) * tgt_inputs_unnormv

                # Define the label for mixed input
                labels_onehot = np.zeros((self.opt.batchSize, self.nclasses + 1), dtype=np.float32)
                for num in range(self.opt.batchSize):
                    labels_onehot[num, src_labels[num]] = mix_ratio
                    labels_onehot[num, self.nclasses] = 1.0 - mix_ratio
                mix_labels_onehot = torch.from_numpy(labels_onehot)

                if self.opt.gpu >= 0:
                    mix_labels_onehot = mix_labels_onehot.cuda()
                mix_labels_onehotv = Variable(mix_labels_onehot)

                # Generating images for both domains (add mixed images)

                src_emb, src_mn, src_sd = self.netF(src_inputsv)
                tgt_emb, tgt_mn, tgt_sd = self.netF(tgt_inputsv)

                # Generate mean and std for mixed samples
                mix_mn = src_mn * mix_ratio + tgt_mn * (1.0 - mix_ratio)
                mix_sd = src_sd * mix_ratio + tgt_sd * (1.0 - mix_ratio)

                src_mn_sd = torch.cat((src_mn, src_sd), 1)
                outC_src_logit, outC_src = self.netC(src_mn_sd)

                src_emb_cat = torch.cat((src_mn, src_sd, src_labels_onehotv), 1)
                src_gen = self.netG(src_emb_cat)

                tgt_emb_cat = torch.cat((tgt_mn, tgt_sd, tgt_labels_onehotv), 1)
                tgt_gen = self.netG(tgt_emb_cat)

                mix_emb_cat = torch.cat((mix_mn, mix_sd, mix_labels_onehotv), 1)
                mix_gen = self.netG(mix_emb_cat)

                # Updating D network

                self.netD.zero_grad()

                src_realoutputD_s, src_realoutputD_c, src_realoutputD_t = self.netD(src_inputs_unnormv)
                errD_src_real_s = self.criterion_s(src_realoutputD_s, reallabelv)
                errD_src_real_c = self.criterion_c(src_realoutputD_c, src_labelsv)

                src_fakeoutputD_s, src_fakeoutputD_c, _ = self.netD(src_gen)
                errD_src_fake_s = self.criterion_s(src_fakeoutputD_s, fakelabelv)

                tgt_realoutputD_s, tgt_realoutputD_c, tgt_realoutputD_t = self.netD(tgt_inputs_unnormv)
                tgt_fakeoutputD_s, tgt_fakeoutputD_c, _ = self.netD(tgt_gen)
                errD_tgt_fake_s = self.criterion_s(tgt_fakeoutputD_s, fakelabelv)

                mix_s, _, mix_t = self.netD(mix_samples)
                if (mix_ratio > 0.5):
                    tmp_margin = 2 * mix_ratio - 1.
                    errD_mix_t = F.triplet_margin_loss(mix_t, src_realoutputD_t, tgt_realoutputD_t, margin=tmp_margin)
                else:
                    tmp_margin = 1. - 2 * mix_ratio
                    errD_mix_t = F.triplet_margin_loss(mix_t, tgt_realoutputD_t, src_realoutputD_t, margin=tmp_margin)
                errD_mix_s = self.criterion_s(mix_s, mix_labelv)
                errD_mix = errD_mix_s + errD_mix_t

                mix_gen_s, _, _ = self.netD(mix_gen)
                errD_mix_gen = self.criterion_s(mix_gen_s, fakelabelv)

                errD = errD_src_real_c + errD_src_real_s + errD_src_fake_s + errD_tgt_fake_s + errD_mix + errD_mix_gen
                errD.backward(retain_graph=True)
                self.optimizerD.step()

                # Updating G network

                self.netG.zero_grad()

                src_fakeoutputD_s, src_fakeoutputD_c, _ = self.netD(src_gen)
                errG_src_c = self.criterion_c(src_fakeoutputD_c, src_labelsv)
                errG_src_s = self.criterion_s(src_fakeoutputD_s, reallabelv)

                mix_gen_s, _, _ = self.netD(mix_gen)
                errG_mix_gen_s = self.criterion_s(mix_gen_s, reallabelv)

                errG = errG_src_c + errG_src_s + errG_mix_gen_s
                errG.backward(retain_graph=True)
                self.optimizerG.step()

                # Updating C network

                self.netC.zero_grad()
                errC = self.criterion_c(outC_src_logit, src_labelsv)
                errC.backward(retain_graph=True)
                self.optimizerC.step()

                # Updating F network

                self.netF.zero_grad()
                err_KL_src = torch.mean(0.5 * torch.sum(torch.exp(src_sd) + src_mn ** 2 - 1. - src_sd, 1))
                err_KL_tgt = torch.mean(0.5 * torch.sum(torch.exp(tgt_sd) + tgt_mn ** 2 - 1. - tgt_sd, 1))
                err_KL = (err_KL_src + err_KL_tgt) * (self.opt.KL_weight)

                errF_fromC = self.criterion_c(outC_src_logit, src_labelsv)

                src_fakeoutputD_s, src_fakeoutputD_c, _ = self.netD(src_gen)
                errF_src_fromD = self.criterion_c(src_fakeoutputD_c, src_labelsv) * (self.opt.adv_weight)

                tgt_fakeoutputD_s, tgt_fakeoutputD_c, _ = self.netD(tgt_gen)
                errF_tgt_fromD = self.criterion_s(tgt_fakeoutputD_s, reallabelv) * (
                        self.opt.adv_weight * self.opt.gamma)

                mix_gen_s, _, _ = self.netD(mix_gen)
                errF_mix_fromD = self.criterion_s(mix_gen_s, reallabelv) * (self.opt.adv_weight * self.opt.delta)

                errF = err_KL + errF_fromC + errF_src_fromD + errF_tgt_fromD + errF_mix_fromD
                errF.backward()
                self.optimizerF.step()

                curr_iter += 1

                # print training information
                if ((i + 1) % 50 == 0):
                    text_format = 'epoch: {}, iteration: {}, errD: {}, errG: {}, ' \
                                  + 'errC: {}, errF: {}'
                    train_text = text_format.format(epoch + 1, i + 1, \
                                                    errD.item(), errG.item(), errC.item(), errF.item())
                    print(train_text)

                # Visualization
                if i == 1:
                    vutils.save_image((src_gen.data / 2) + 0.5,
                                      '%s/source_generation/source_gen_%d.png' % (self.opt.outf, epoch))
                    vutils.save_image((tgt_gen.data / 2) + 0.5,
                                      '%s/target_generation/target_gen_%d.png' % (self.opt.outf, epoch))
                    vutils.save_image((mix_gen.data / 2) + 0.5,
                                      '%s/mix_generation/mix_gen_%d.png' % (self.opt.outf, epoch))
                    vutils.save_image((mix_samples.data / 2) + 0.5,
                                      '%s/mix_images/mix_samples_%d.png' % (self.opt.outf, epoch))

                # Learning rate scheduling
                if self.opt.lrd:
                    self.optimizerD = utils.exp_lr_scheduler(self.optimizerD, epoch, self.opt.lr, self.opt.lrd,
                                                             curr_iter)
                    self.optimizerF = utils.exp_lr_scheduler(self.optimizerF, epoch, self.opt.lr, self.opt.lrd,
                                                             curr_iter)
                    self.optimizerC = utils.exp_lr_scheduler(self.optimizerC, epoch, self.opt.lr, self.opt.lrd,
                                                             curr_iter)

                    # Validate every epoch
            self.validate(epoch + 1)


class Sourceonly(object):

    def __init__(self, opt, nclasses, source_trainloader, target_valloader):

        self.source_trainloader = source_trainloader
        self.target_valloader = target_valloader
        self.opt = opt
        self.best_val = 0

        # Defining networks and optimizers
        self.nclasses = nclasses
        self.netF = models.netF(opt)
        self.netC = models.netC(opt, nclasses)

        # Weight initialization
        self.netF.apply(utils.weights_init)
        self.netC.apply(utils.weights_init)

        # Defining loss criterions
        self.criterion = nn.CrossEntropyLoss()

        if opt.gpu >= 0:
            self.netF.cuda()
            self.netC.cuda()
            self.criterion.cuda()

        # Defining optimizers
        self.optimizerF = optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(0.8, 0.999))
        self.optimizerC = optim.Adam(self.netC.parameters(), lr=opt.lr, betas=(0.8, 0.999))

    """
    Validation function
    """

    def validate(self, epoch):

        self.netF.eval()
        self.netC.eval()
        total = 0
        correct = 0

        # Testing the model
        for i, datas in enumerate(self.target_valloader):
            inputs, labels = datas
            inputv, labelv = Variable(inputs.cuda()), Variable(labels.cuda())

            outC = self.netC(self.netF(inputv))
            _, predicted = torch.max(outC.data, 1)
            total += labels.size(0)
            correct += ((predicted == labels.cuda()).sum())

        val_acc = 100 * float(correct) / total

        # Saving checkpoints
        if val_acc > self.best_val:
            self.best_val = val_acc
            torch.save(self.netF.state_dict(), '%s/models/model_best_netF_sourceonly.pth' % (self.opt.outf))
            torch.save(self.netC.state_dict(), '%s/models/model_best_netC_sourceonly.pth' % (self.opt.outf))

        print('%s| Epoch: %d, Val Accuracy: %f, Best Accuracy: %f %%\n' % (
            datetime.datetime.now(), epoch, val_acc, self.best_val))

    """
    Train function
    """

    def train(self):

        curr_iter = 0
        for epoch in range(self.opt.nepochs):

            self.netF.train()
            self.netC.train()

            for i, datas in enumerate(self.source_trainloader):

                ###########################
                # Forming input variables
                ###########################

                src_inputs, src_labels = datas
                if self.opt.gpu >= 0:
                    src_inputs, src_labels = src_inputs.cuda(), src_labels.cuda()
                src_inputsv, src_labelsv = Variable(src_inputs), Variable(src_labels)

                ###########################
                # Updates
                ###########################

                self.netC.zero_grad()
                self.netF.zero_grad()
                outC = self.netC(self.netF(src_inputsv))
                loss = self.criterion(outC, src_labelsv)
                loss.backward()
                self.optimizerC.step()
                self.optimizerF.step()

                curr_iter += 1

                # print training information
                if ((i + 1) % 50 == 0):
                    text_format = 'epoch: {}, iteration: {}, errC: {}'
                    train_text = text_format.format(epoch + 1, i + 1, loss.item())
                    print(train_text)

                # Learning rate scheduling
                if self.opt.lrd:
                    self.optimizerF = utils.exp_lr_scheduler(self.optimizerF, epoch, self.opt.lr, self.opt.lrd,
                                                             curr_iter)
                    self.optimizerC = utils.exp_lr_scheduler(self.optimizerC, epoch, self.opt.lr, self.opt.lrd,
                                                             curr_iter)

                    # Validate every epoch
            self.validate(epoch)
