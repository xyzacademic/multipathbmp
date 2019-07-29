import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        # print(m)
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm3d):
        # print(m)
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def conv_blocks(channels=32, n_layers=1, kernel_size=3, activation=nn.ReLU(inplace=True)):
    '''

    :param channels: the num of filters
    :param kernel_size: kernel size
    :param n_layers: the number of conv layers repeating
    :param activation: activation function
    :return: value unactivated

    blocks: [activation->convolution->bn]

    '''

    layers = []
    for i in range(n_layers):
        layers += [
            activation,
            nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(channels),
        ]

    return nn.Sequential(*layers)


class Encoder(nn.Module):
    '''
    Encoder for shape (192 * 160)
    '''

    def __init__(self):
        super(Encoder, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(2, 2)
        # self.input_bn = nn.BatchNorm2d(1)
        self.conv0 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(32)
        self.conv1 = conv_blocks(channels=32, n_layers=1, kernel_size=3, activation=self.act)
        self.pool1 = self.pool
        self.conv2 = conv_blocks(channels=32, n_layers=3, kernel_size=3, activation=self.act)
        self.pool2 = self.pool
        self.conv3 = conv_blocks(channels=32, n_layers=3, kernel_size=3, activation=self.act)
        self.pool3 = self.pool
        self.conv4 = conv_blocks(channels=32, n_layers=3, kernel_size=3, activation=self.act)
        self.pool4 = self.pool
        self.conv5 = conv_blocks(channels=32, n_layers=2, kernel_size=3, activation=self.act)

        self.apply(_weights_init)

    def forward(self, x):
        # x = self.input_bn(x)
        out = self.conv0(x)
        out = self.bn0(out)
        conv1 = self.conv1(out)
        out = self.pool1(conv1)
        conv2 = self.conv2(out)
        out = self.pool2(conv2)
        conv3 = self.conv3(out)
        out = self.pool3(conv3)
        conv4 = self.conv4(out)
        out = self.pool4(conv4)
        conv5 = self.conv5(out)

        return (conv1, conv2, conv3, conv4, conv5)


class conv_fuse(nn.Module):
    '''
    Combine different mode's feature throught 3D convolution
    '''

    def __init__(self, activation):
        super(conv_fuse, self).__init__()
        self.activation = activation
        self.conv = nn.Conv3d(32, 32, kernel_size=(2, 1, 1), bias=False)
        self.bn = nn.BatchNorm3d(32)
        self.apply(_weights_init)

    def forward(self, x):
        out = self.activation(x)
        out = self.conv(out)
        out = self.bn(out)
        out = out.squeeze()
        return out


class Decoder(nn.Module):
    '''
    Decoder for shape: (12, 10), (24, 20), (48, 40), (96, 80)
    '''

    def __init__(self):
        super(Decoder, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, bias=False)
        self.conv1 = conv_blocks(channels=32, n_layers=2, kernel_size=3, activation=self.act)
        self.drop1 = nn.Dropout2d(0.2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, bias=False)
        self.conv2 = conv_blocks(channels=32, n_layers=2, kernel_size=3, activation=self.act)
        self.drop2 = nn.Dropout2d(0.2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, bias=False)
        self.conv3 = conv_blocks(channels=32, n_layers=2, kernel_size=3, activation=self.act)
        self.drop3 = nn.Dropout2d(0.2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, bias=False)
        self.conv4 = conv_blocks(channels=32, n_layers=2, kernel_size=3, activation=self.act)
        self.drop4 = nn.Dropout2d(0.2)
        self.conv = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.apply(_weights_init)

    def forward(self, features):
        assert len(features) == 5
        f4, f3, f2, f1, f0 = features  # inverse variables' name order

        out1 = self.deconv1(f0)
        out1 = f1 + out1  # multiply encoder's feature with related output of deconvolution layer.
        out1 = self.conv1(out1)
        out1 = self.drop1(out1)

        out2 = self.deconv2(out1)
        out2 = f2 + out2
        out2 = self.conv2(out2)
        out2 = self.drop2(out2)

        out3 = self.deconv3(out2)
        out3 = f3 + out3
        out3 = self.conv3(out3)
        out3 = self.drop3(out3)

        out4 = self.deconv4(out3)
        out4 = f4 + out4
        out4 = self.conv4(out4)
        out4 = self.drop4(out4)
        out = self.conv(out4)

        return out

class ModalilyFuse(nn.Module):
    def __init__(self):
        super(ModalilyFuse, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.fuse0 = conv_fuse(self.act)
        self.fuse1 = conv_fuse(self.act)
        self.fuse2 = conv_fuse(self.act)
        self.fuse3 = conv_fuse(self.act)
        self.fuse4 = conv_fuse(self.act)
        self.apply(_weights_init)

    def forward(self, anat_features, flair_features):
        f0 = self.fuse0(torch.stack([anat_features[0], flair_features[0]], dim=2))
        f1 = self.fuse0(torch.stack([anat_features[1], flair_features[1]], dim=2))
        f2 = self.fuse0(torch.stack([anat_features[2], flair_features[2]], dim=2))
        f3 = self.fuse0(torch.stack([anat_features[3], flair_features[3]], dim=2))
        f4 = self.fuse0(torch.stack([anat_features[4], flair_features[4]], dim=2))
        features = (f0, f1, f2, f3, f4)

        return features

class DTS(nn.Module):
    '''
    Net work used to do segmentation.
    Input shape: (batch_size, mode, 192, 160)
    Output shape:
    '''

    def __init__(self):
        super(DTS, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.anat_encoder = Encoder()
        self.flair_encoder = Encoder()
        self.fuse = ModalilyFuse()

        self.decoder = Decoder()

        self.apply(_weights_init)

    def forward(self, x):
        x_anat = x[:, 0:1, :, :]
        x_flair = x[:, 1:2, :, :]
        anat_features = self.anat_encoder(x_anat)
        flair_features = self.flair_encoder(x_flair)
        features = self.fuse(anat_features, flair_features)

        out = self.decoder(features)

        return out

class Combine(nn.Module):
    def __init__(self, norm='BatchNorm'):
        super(Combine, self).__init__()
        self.norm = nn.BatchNorm3d if norm == 'BatchNorm' else nn.InstanceNorm3d
        self.activation = F.relu_
        self.conv1 = nn.Conv3d(27, 36, kernel_size=(3, 3, 3), padding=1, bias=False)
        self.bn1 = self.norm(36)
        self.conv2 = nn.Conv3d(36, 9, kernel_size=(3, 3, 3), padding=1, bias=False)
        self.bn2 = self.norm(9)
        self.conv_shortcut = nn.Conv3d(27, 9, kernel_size=(1, 1, 1), padding=0, bias=False)
        self.bn_shorcut = self.norm(9)
        self.conv3 = nn.Conv3d(9, 2, kernel_size=(3, 3, 3), padding=1, bias=False)
        self.apply(_weights_init)

    def forward(self, x):
        # assert x.size() == (1, 18, 160, 192, 160)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.conv_shortcut(x)
        shortcut = self.bn_shorcut(shortcut)
        out = out + shortcut
        out = self.activation(out)
        # out = out.sigmoid_()
        out = self.conv3(out)
        return out





class OneModel(nn.Module):
    def __init__(self, mode=1):
        super(OneModel, self).__init__()
        self.mode = mode
        self.encoder = Encoder()
        self.dense = nn.Linear(32 * 5 if self.mode == 5 else 32, 100)
        # self.bn = nn.BatchNorm1d(100)
        self.linear_rotation = nn.Linear(32 * 5 if self.mode == 5 else 32, 4)
        self.linear_position = nn.Linear(32 * 5 if self.mode == 5 else 32, 6)

    def forward(self, x):
        features = self.encoder(x)
        if self.mode == 5:
            vectors = [F.avg_pool2d(features[i], kernel_size=features[i].size(-1)).squeeze() for i in range(5)]
            out = torch.cat(vectors, dim=1)
        elif self.mode == 1:
            out = F.avg_pool2d(features[4], kernel_size=features[4].size(3), stride=1).squeeze()
        # out = self.dense(out)
        # out = self.bn(out)
        # out = F.relu(out)
        rotation = self.linear_rotation(out)
        position = self.linear_position(out)

        return rotation, position


if __name__ == '__main__':
    x = torch.randn((32, 2, 160, 160)).cuda()

    net = DTS()
    net.cuda()
    y = net(x)
