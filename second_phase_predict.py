import torch
import argparse
import numpy as np
from time import time
import os
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils import Preprocessing, get_data, Model, dice_coef, IOU, record_csv, dice_loss
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
parser.add_argument('--epoch', default=50, type=int, help='Initial learning rate.')
parser.add_argument('--gpu', default=-1, type=int, help='Using which gpu.')
parser.add_argument('--threshold', default=0.9, type=float, help='Threshold')
# parser.add_argument('--epoch', default=50, type=int, help='Initial learning rate.')
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
config['wd'] = 0.0001
config['epoch'] = args.epoch
config['lr_decay'] = np.arange(2, 50)
config['experiment_name'] = args.net
config['save_prediction_path'] = 'final_labels'
data_path = '%s/second_data%s' % (args.net, args.data)
train_data, train_label, test_data, test_label = get_data(data_path)
# del test_data, test_label
torch.manual_seed(config['seed'])

if not os.path.exists(config['experiment_name']):
    os.mkdir(config['experiment_name'])

os.chdir(config['experiment_name'])

c = torch.from_numpy(test_data)
testset = torch.utils.data.TensorDataset(c)


test_loader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers=1)

net = Combine(norm='InstanceNorm')
model = Model(net=net, config=config)


model.resume(save_path=config['save_path'], filename='ckpt_%d.t7' % config['epoch'])

print('Test inference:')
test_images = model.inference(test_loader)


if not os.path.exists(config['save_prediction_path']):
    os.mkdir(config['save_prediction_path'])
os.chdir(config['save_prediction_path'])
print('Save images...')
np.save("%s.npy" % (args.data), test_images.astype(np.uint8))
print(test_images.shape)
os.chdir(os.pardir)
os.chdir(os.pardir)
