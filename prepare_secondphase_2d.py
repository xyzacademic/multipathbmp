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
config['resume'] = True
config['use_cuda'] = True
config['fp16'] = False
config['dtype'] = torch.float16 if config['fp16'] else torch.float32
config['gpu'] = args.gpu
config['batch_size'] = 64
config['seed'] = 2018
config['save_path'] = "checkpoints/%s_%s_%s" % (args.data, config['view'], config['norm_axis'])
# config['lr'] = args.lr
config['wd'] = 0.0001
config['epoch'] = args.epoch
# config['lr_decay'] = np.arange(2, 50)
config['experiment_name'] = args.net
config['save_prediction_path'] = 'second_phase_data'

data_path = 'data%s' % args.data
train_data, train_label, test_data, test_label = get_data(data_path)
# del test_data, test_label

if not os.path.exists(config['experiment_name']):
    os.mkdir(config['experiment_name'])

os.chdir(config['experiment_name'])

pre = Preprocessing(config['view'], normalize_mode=config['norm_axis'])

c = pre.transform_data(test_data, normalize=True)

c = torch.from_numpy(c)
testset = torch.utils.data.TensorDataset(c)


test_loader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers=1)

if args.net == 'ours':
    net = DTS()
elif args.net == 'unet':
    net = UNet(n_channels=2, n_classes=2)
elif args.net == 'uresnet':
    net = UResNet(num_classes=2, input_channels=2, inplanes=16)

model = Model(net=net, config=config)

save_path = config['save_path']


model.resume(save_path=config['save_path'], filename='ckpt_%d.t7' % config['epoch'])

print('Prepare data for second phase training:')
test_images = model.prepare_second_phase_data(test_loader)
test_images = pre.correction_data(test_images)

if not os.path.exists(config['save_prediction_path']):
    os.mkdir(config['save_prediction_path'])
os.chdir(config['save_prediction_path'])
print('Save images...')
np.save("%s_%s_%s.npy" % (args.data, config['view'], config['norm_axis']), test_images)
print(test_images.shape)
os.chdir(os.pardir)
os.chdir(os.pardir)