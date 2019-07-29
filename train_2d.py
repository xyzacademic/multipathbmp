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
from net import DTS
from base_model import UResNet, UNet
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
parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate.')
parser.add_argument('--epoch', default=50, type=int, help='Initial learning rate.')
parser.add_argument('--gpu', default=-1, type=int, help='Using which gpu.')
parser.add_argument('--threshold', default=0.9, type=float, help='Threshold')
parser.add_argument('--net', default='ours', type=str, help='which network')
args = parser.parse_args()


####
# Global Flag
###

config = {}

# Config setting
config['view'] = args.view
config['norm_axis'] = args.norm_axis
config['resume'] = False
config['use_cuda'] = True
config['fp16'] = False
config['dtype'] = torch.float16 if config['fp16'] else torch.float32
config['gpu'] = args.gpu
config['batch_size'] = 32
config['seed'] = 2018
config['save_path'] = "checkpoints/%s_%s_%s" % (args.data, config['view'], config['norm_axis'])
config['lr'] = args.lr
config['wd'] = 0.0001
config['epoch'] = args.epoch
config['lr_decay'] = np.arange(2, config['epoch'])
config['experiment_name'] = args.net

data_path = 'data%s' % args.data
train_data, train_label, test_data, test_label = get_data(data_path)
# del test_data, test_label
torch.manual_seed(config['seed'])

if not os.path.exists(config['experiment_name']):
    os.mkdir(config['experiment_name'])

os.chdir(config['experiment_name'])

pre = Preprocessing(config['view'], normalize_mode=config['norm_axis'])
a, b = pre.transform_pair(train_data, train_label, normalize=True)
c = pre.transform_data(test_data, normalize=True)
index = b.max(axis=(1, 2)) == 1

trainset = torch.utils.data.TensorDataset(torch.from_numpy(a[index]), torch.from_numpy(b[index]))
a, b, c = torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(c)
trainset2 = torch.utils.data.TensorDataset(a, b)
testset = torch.utils.data.TensorDataset(c)
valset = torch.utils.data.TensorDataset(a)


train_loader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=1)
train_loader2 = torch.utils.data.DataLoader(trainset2, batch_size=config['batch_size'], shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(valset, batch_size=192, shuffle=False, num_workers=1)
test_loader = torch.utils.data.DataLoader(testset, batch_size=192, shuffle=False, num_workers=1)

if args.net == 'ours':
    net = DTS()
elif args.net == 'unet':
    net = UNet(n_channels=2, n_classes=2)
elif args.net == 'uresnet':
    net = UResNet(num_classes=2, input_channels=2, inplanes=16)

model = Model(net=net, config=config)
model.optimizer_initialize()
# weight = torch.Tensor([0.509, 28.16])
# loss = nn.CrossEntropyLoss(weight=weight)
loss = DiceLoss(reduce='mean')
model.loss_initialize(loss)
# model.training_mode(train_loader)
# w1 = 28.16

start_epoch = 1
save_path = config['save_path']
val_dice = []
test_dice = []
for epoch in range(start_epoch, start_epoch + config['epoch']):

    if epoch < 26:
        model.train(epoch, train_loader)
    else:
        model.train(epoch, train_loader2)
    model.save(save_path, 'ckpt_%d.t7' % epoch)
    if epoch in config['lr_decay']:
        model.optimizer.param_groups[0]['lr'] *= 0.91

    # model.resume(save_path=config['save_path'], filename='ckpt_%d.t7' % epoch)

#     print('Validation inference:')
#     val_images = model.inference(val_loader)
#     val_dice.append(model.evaluate(val_images, train_label, dice_coef, pre))
#     print('Test inference:')
#     test_images = model.inference(test_loader)
#     test_dice.append(model.evaluate(test_images, test_label, dice_coef, pre))
#
#
# if not os.path.exists('results%s' % args.data):
#     os.mkdir('results%s' % args.data)
# record_csv('results%s/Train_%s_%s.csv' % (args.data, config['view'], config['norm_axis']), val_dice, '../statistic/train_%s.csv'%args.data)
# record_csv('results%s/Test_%s_%s.csv' % (args.data, config['view'], config['norm_axis']), test_dice, '../statistic/test_%s.csv'%args.data)

os.chdir(os.pardir)