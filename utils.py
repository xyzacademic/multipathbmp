import numpy as np
import os
from time import time
import torch 
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import pandas as pd


'''
Create some class here
'''


class Preprocessing(object):
    def __init__(self, data_mode='XY', normalize_mode='3'):
        self.data_mode = data_mode
        self.normalize_mode = normalize_mode
    
    def transform_pair(self, data, label, normalize=False):
        '''
        
        Transform data and label based on data_mode
        '''
        train_data = data
        train_label = label
        assert train_data.shape == (train_data.shape[0], 2, 192, 224, 192)
        assert train_label.shape == (train_label.shape[0], 192, 224, 192)


        if self.data_mode == 'XY':
            train_data = train_data.transpose((0, 2, 1, 3, 4)).reshape((-1, 2, 224, 192))
            train_label = train_label.reshape((-1, 224, 192))
            train_data[:, 1:2:, :] = train_data[:, 0:1, :, ::-1]


        elif self.data_mode =='ZY':
            train_data = train_data.transpose((0, 3, 1, 2, 4)).reshape((-1, 2, 192, 192))
            train_label = train_label.transpose((0, 2, 1, 3)).reshape((-1, 192, 192))
            train_data[:, 1:2:, :] = train_data[:, 0:1, :, ::-1]


        elif self.data_mode == 'ZX':
            train_data = train_data.transpose((0, 4, 1, 2, 3)).reshape((-1, 2, 192, 224))
            train_label = train_label.transpose((0, 3, 1, 2)).reshape((-1, 192, 224))

        else:
            raise "Does not support %s mode"%self.data_mode

        if normalize == True:
            train_data = self.normalize(train_data)

        return train_data, train_label


    def transform_data(self, data, normalize=False):
        '''

        Transform data based on data_mode
        '''
        train_data = data

        assert train_data.shape == (train_data.shape[0], 2, 192, 224, 192)

        if self.data_mode == 'XY':
            train_data = train_data.transpose((0, 2, 1, 3, 4)).reshape((-1, 2, 224, 192))
            train_data[:, 1:2:, :] = train_data[:, 0:1, :, ::-1]

        elif self.data_mode == 'ZY':
            train_data = train_data.transpose((0, 3, 1, 2, 4)).reshape((-1, 2, 192, 192))
            train_data[:, 1:2:, :] = train_data[:, 0:1, :, ::-1]

        elif self.data_mode == 'ZX':
            train_data = train_data.transpose((0, 4, 1, 2, 3)).reshape((-1, 2, 192, 224))

        else:
            raise "Does not support %s mode" % self.data_mode

        if normalize == True:
            train_data = self.normalize(train_data)

        return train_data

    def normalize(self, data):
        '''

        Normalize data based on normalize_mode
        '''

        assert len(data.shape) == 4
        if self.normalize_mode == '12':
            mean = data.mean(axis=(2, 3), dtype=np.float32, keepdims=True)
            std = data.std(axis=(2, 3), dtype=np.float32, keepdims=True)
            data = np.nan_to_num((data - mean)/std)

        elif self.normalize_mode == '3':
            shape = data.shape
            temp_data = data.reshape((-1, (192*224*192)//data.shape[2]//data.shape[3], 2, data.shape[2], data.shape[3]))
            mean = temp_data.mean(axis=1, dtype=np.float32, keepdims=True)
            std = temp_data.std(axis=1, dtype=np.float32, keepdims=True)
            data = np.nan_to_num((temp_data - mean)/std).reshape(shape)

        elif self.normalize_mode == '123':
            shape = data.shape
            temp_data = data.reshape((-1, (192*224*192)//data.shape[2]//data.shape[3], 2, data.shape[2], data.shape[3]))
            mean = temp_data.mean(axis=1, dtype=np.float32, keepdims=True)
            std = temp_data.std(axis=1, dtype=np.float32, keepdims=True)
            data = np.nan_to_num((temp_data - mean) / std).reshape(shape)
            mean = data.mean(axis=(2, 3), dtype=np.float32, keepdims=True)
            std = data.std(axis=(2, 3), dtype=np.float32, keepdims=True)
            data = np.nan_to_num((data - mean)/std)

        return data

    def correction_label(self, label):
        '''

        Correct label to original shape (192, 224, 192)
        '''

        assert len(label.shape) == 3
        if self.data_mode == 'XY':
            assert label.shape[1:] == (224, 192)
            label = label.reshape((-1, 192, 224, 192))

        elif self.data_mode == 'ZY':
            assert label.shape[1:] == (192, 192)
            label = label.reshape((-1, 224, 192, 192)).transpose((0, 2, 1, 3))

        elif self.data_mode == 'ZX':
            assert label.shape[1:] == (192, 224)
            label = label.reshape((-1, 192, 192, 224)).transpose((0, 2, 3, 1))

        return label



    def correction_data(self, data):
        '''

        Correct label to original shape (2, 192, 224, 192)
        '''

        assert len(data.shape) == 4
        channels = 3
        if self.data_mode == 'XY':
            assert data.shape[1:] == (channels, 224, 192)
            data = data.reshape((-1, 192, channels, 224, 192)).transpose((0, 2, 1, 3, 4))

        elif self.data_mode == 'ZY':
            assert data.shape[1:] == (channels, 192, 192)
            data = data.reshape((-1, 224, channels, 192, 192)).transpose((0, 2, 3, 1, 4))

        elif self.data_mode == 'ZX':
            assert data.shape[1:] == (channels, 192, 224)
            data = data.reshape((-1, 192, channels, 192, 224)).transpose((0, 2, 3, 4, 1))

        return data

def get_data(data_path):
    """

    :param data_path:
    :return:
    """
    curdir = os.getcwd()
    os.chdir(data_path)
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    train_label = np.load('train_label.npy')
    test_label = np.load('test_label.npy')
    os.chdir(curdir)

    return train_data, train_label, test_data, test_label


class Model(object):
    def __init__(self, net=None, config=None):
        self.train_loader = None
        self.test_loader = None
        self.net = net
        self.config = config
        # assert isinstance(self.net, nn.Module)
        self.net_initialize()
        self.optimizer = None

    def net_initialize(self):
        config = self.config
        if config['use_cuda'] is True:
            print('start move to cuda')
            torch.cuda.manual_seed_all(config['seed'])
            cudnn.benchmark = True
            if config['fp16'] is True:
                self.net.half()
                for layer in self.net.modules():
                    if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm3d):
                        layer.float()
            if config['gpu'] == -1:
                self.net = torch.nn.DataParallel(self.net, device_ids=[0, 1])
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cuda:%d" % config['gpu'])
            self.net.to(device=self.device)

    def optimizer_initialize(self, params_list=None):
        config = self.config
        params = self.net.parameters() if params_list is None else params_list
        self.optimizer = optim.SGD(
            params,
            lr=config['lr'],
            momentum=0.9,
            weight_decay=config['wd'],
            nesterov=True
        )

    def loss_initialize(self, loss):
        config = self.config
        self.criterion = loss
        if isinstance(loss, nn.CrossEntropyLoss):
            if config['use_cuda'] is True:
                self.criterion.to(device=self.device, dtype=config['dtype'])

    def resume(self, save_path, filename):
        assert os.path.exists(save_path)
        print('==> Resuming from checkpoint..')
        path = os.path.join(save_path, filename)
        checkpoint = torch.load(path)
        if self.config['gpu'] == -1:
            self.net.module.load_state_dict(checkpoint['net'])
        else:
            self.net.load_state_dict(checkpoint['net'])

    def save(self, save_path, filename):
        state_dict = self.net.module.state_dict() if self.config['gpu'] == -1 else self.net.state_dict()
        print('Saving...')
        state = {
            'net': state_dict
        }
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        path = os.path.join(save_path, filename)
        torch.save(state, path)

    def training_mode(self, train_loader):
        save_path = self.config['save_path']
        start_epoch = 1
        for epoch in range(start_epoch, start_epoch + self.config['epoch']):
            self.train(epoch, train_loader)
            self.save(save_path, 'ckpt_%d.t7' % epoch)
            if epoch in self.config['lr_decay']:
                self.optimizer.param_groups[0]['lr'] *= 0.1

    def train(self, epoch=0, train_loader=None):
        config = self.config
        self.train_loader = train_loader
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        start = time()

        assert self.train_loader is not None
        assert self.net is not None
        assert self.optimizer is not None
        assert self.criterion is not None

        for batch_idx, (data, target) in enumerate(self.train_loader):
            if config['use_cuda'] is True:
                data, target = data.to(device=self.device, dtype=self.config['dtype']), target.to(device=self.device)

            self.optimizer.zero_grad()
            outputs = self.net(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * target.size(0)

        train_loss = train_loss / len(self.train_loader.dataset)
        print('Train loss: %.5f' % train_loss)
        print('This epoch cost %.2f seconds' % (time() - start))
        print('Current learning rate: %.5f' % self.optimizer.param_groups[0]['lr'])


    def inference(self, test_loader=None):
        config = self.config
        self.test_loader = test_loader
        assert self.test_loader is not None
        assert self.net is not None

        self.net.eval()
        start = time()
        images = []

        with torch.no_grad():
            for batch_idx, (data,) in enumerate(self.test_loader):
                if config['use_cuda'] is True:
                    data = data.to(device=self.device, dtype=self.config['dtype'])

                outputs = self.net(data)
                pre = outputs.max(1)[1]
                images.append(pre.data.cpu())

        images = torch.cat(images, dim=0)
        print('This inference cost %.2f seconds' % (time() - start))

        return images.numpy()

    def prepare_second_phase_data(self, test_loader=None):
        config = self.config
        self.test_loader = test_loader
        assert self.test_loader is not None
        assert self.net is not None

        self.net.eval()
        start = time()
        images = []

        with torch.no_grad():
            for batch_idx, (data,) in enumerate(self.test_loader):
                if config['use_cuda'] is True:
                    data = data.to(device=self.device, dtype=self.config['dtype'])

                outputs = self.net(data)
                pre = torch.cat([outputs.max(1)[1].unsqueeze(dim=1).to(dtype=self.config['dtype']), data], dim=1) #concat mask
                # pre = outputs.max(1)[1].unsqueeze(dim=1).to(dtype=self.config['dtype']) * data
                # pre = torch.cat([F.softmax(outputs, dim=1)[:, 1:2, :, :], data], dim=1)
                # pre = data
                images.append(pre.data.cpu())

        images = torch.cat(images, dim=0)
        print('This prepare data phase cost %.2f seconds' % (time() - start))

        return images.numpy()


    def evaluate(self, pred, label, eval_func, pre):

        prediction = pre.correction_label(pred)
        evaluation = eval_func(prediction, label)

        return evaluation

    def evaluate_2p(self, pred, label, eval_func):

        evaluation = eval_func(pred, label)

        return evaluation

def dice_coef(pred, target):
    assert pred.shape == target.shape
    a = pred + target
    overlap = (pred * target).sum(axis=(1, 2, 3)) * 2
    union = a.sum(axis=(1, 2, 3))
    epsilon = 0.0001
    dice = overlap / (union + epsilon)

    return dice

def IOU(pred, target):
    assert pred.shape == target.shape
    a = pred + target
    overlap = (pred * target).sum(axis=(1, 2, 3))
    union = (a > 0).sum(axis=(1, 2, 3))
    epsilon = 0.0001
    iou = overlap / (union + epsilon)

    return iou

def record_csv(filename, record_list, preload):
    df = pd.read_csv(preload)
    name = df['Name'].tolist()
    del df
    columns = ['Epoch'] + name
    df = pd.DataFrame(data=[], index=np.arange(len(record_list)),columns=columns)
    for i in range(len(record_list)):
        df.iloc[i] = [i+1] + record_list[i].tolist()
    df.to_csv(filename, index=False)
    print('Save results to %s' % filename)


def dice_loss(input, target):
    smooth = 1.
    # pred = input.max(1)[1]
    input = F.softmax(input, dim=1)[:, 1]
    iflat = input.contiguous().view(-1)
    tflat = target.float().contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        # if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        # if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target.long())
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()