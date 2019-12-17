import torch
from torch.nn import init

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		size = m.weight.size()
		m.weight.data.normal_(0.0, 0.1)
		m.bias.data.fill_(0)

def weights_init_xavier(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.xavier_normal(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal(m.weight.data, 1.0, 0.02)
		init.constant(m.bias.data, 0.0)

def lr_scheduler(optimizer, lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return optimizer

def exp_lr_scheduler(optimizer, epoch, init_lr, lrd, nevals):
    """Implements torch learning reate decay with SGD"""
    lr = init_lr / (1 + nevals*lrd)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
