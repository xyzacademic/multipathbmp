import torch
import argparse
import numpy as np
from time import time
import os
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils import Preprocessing, get_data, Model, dice_coef, IOU, record_csv
from lossfunction import DiceLoss, DiceLossStack, FocalLoss, GDL
from net import Combine
import torch.utils.data

parser = argparse.ArgumentParser(description='Train DTS with 2d segmentation')

parser.add_argument('--view', default='XY', type=str, help='View from which side')
parser.add_argument('--norm-axis', default='3', type=str, help='Normalization axis')
parser.add_argument('--data', default=0, type=str, help='Data source')
# parser.add_argument('--fid', default=0, type=int, help='Index of file which features save as.')
# parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode.')
# parser.add_argument('--resume', action='store_true', help='Run model fp16 mode.')
# parser.add_argument('--num-features', default=1000, type=int, help='The number of features.')
# parser.add_argument('--batch-size', default=2, type=int, help='Batch size.')
# parser.add_argument('--seed', default=2018, type=int, help='Random seed.')
parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate.')
parser.add_argument('--gpu', default=-1, type=int, help='Using which gpu.')
parser.add_argument('--threshold', default=0.9, type=float, help='Threshold')
parser.add_argument('--epoch', default=50, type=int, help='Initial learning rate.')
parser.add_argument('--net', default='ours', type=str, help='which network')
args = parser.parse_args()

####
# Global Flag
###

config = {}

# Config setting
# config['view'] = args.view
# config['norm_axis'] = args.norm_axis
config['resume'] = False
config['use_cuda'] = True
config['fp16'] = False
config['dtype'] = torch.float16 if config['fp16'] else torch.float32
config['gpu'] = args.gpu
config['batch_size'] = 2
config['seed'] = 2018
config['save_path'] = "checkpoints_2p/%s" % (args.data)
config['lr'] = args.lr
config['wd'] = 0.0001
config['epoch'] = args.epoch
config['lr_decay'] = np.arange(2, config['epoch'])
config['experiment_name'] = args.net

data_path = '%s/second_data%s' % (args.net, args.data)
train_data, train_label, test_data, test_label = get_data(data_path)
# del test_data, test_label
torch.manual_seed(config['seed'])

if not os.path.exists(config['experiment_name']):
    os.mkdir(config['experiment_name'])

os.chdir(config['experiment_name'])

a, b = torch.from_numpy(train_data), torch.from_numpy(train_label)
# testset = torch.utils.data.TensorDataset(c)
trainset = torch.utils.data.TensorDataset(a, b)
# valset = torch.utils.data.TensorDataset(a)
del train_data, train_label, test_data

train_loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=1)
# val_loader = torch.utils.data.DataLoader(valset, batch_size=config['batch_size'], shuffle=False, num_workers=1)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers=1)

net = Combine(norm='InstanceNorm')
model = Model(net=net, config=config)
model.optimizer_initialize()

loss = DiceLoss(reduce='mean')
model.loss_initialize(loss)

start_epoch = 1
save_path = config['save_path']
val_dice = []
test_dice = []
for epoch in range(start_epoch, start_epoch + config['epoch']):

    model.train(epoch, train_loader)
    model.save(save_path, 'ckpt_%d.t7' % epoch)
    if epoch in config['lr_decay']:
        model.optimizer.param_groups[0]['lr'] *= 0.91

#     print('Validation inference:')
#     val_images = model.inference(val_loader)
#     val_dice.append(model.evaluate_2p(val_images, train_label, dice_coef))
#     print('Test inference:')
#     test_images = model.inference(test_loader)
#     test_dice.append(model.evaluate_2p(test_images, test_label, dice_coef))
#
# if not os.path.exists('results'):
#     os.mkdir('results')
# record_csv('results/Train_%s.csv' % (args.data), val_dice,
#            '../statistic/train_%s.csv' % args.data)
# record_csv('results/Test_%s.csv' % (args.data), test_dice,
#            '../statistic/test_%s.csv' % args.data)

os.chdir(os.pardir)
