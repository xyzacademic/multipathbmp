import torch
import torch.nn as nn
import torch.nn.functional as F



class DiceLossStack(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLossStack, self).__init__()
        self.smooth = smooth

        return

    def forward(self, input, target):
        input_ = F.softmax(input, dim=1)[:, 1]
        iflat = input_.contiguous().view(-1)
        tflat = target.float().contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + self.smooth) /
                    (iflat.sum() + tflat.sum() + self.smooth))


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., reduce='mean'):
        super(DiceLoss, self).__init__()
        self.reduce = reduce
        self.smooth = smooth

        return

    def forward(self, input, target):
        N = target.size(0)
        C = input.size(1)
        labels = target.unsqueeze(dim=1)
        one_hot = torch.zeros_like(input)
        target = one_hot.scatter_(1, labels.data, 1)
        input_ = F.softmax(input, dim=1)
        iflat = input_.contiguous().view(N, C, -1)
        tflat = target.contiguous().view(N, C, -1)
        intersection = (iflat * tflat).sum(dim=2)
        dice = (2. * intersection + self.smooth) / (iflat.sum(dim=2) + tflat.sum(dim=2) + self.smooth)
        if self.reduce == 'mean':
            loss = (C * 1.0 - dice.sum(dim=1)).mean()
        elif self.reduce == 'sum':
            loss = N - dice.sum()

        return loss


class GDL(nn.Module):
    def __init__(self, smooth=0.0001, reduce='mean'):
        super(GDL, self).__init__()
        self.smooth = smooth
        self.reduce = reduce
    def forward(self, input, target):
        N = target.size(0)
        C = input.size(1) - 1
        labels = target.unsqueeze(dim=1)
        one_hot = torch.zeros_like(input)
        target = one_hot.scatter_(1, labels.data, 1)[:, 1:]
        input_ = F.softmax(input, dim=1)[:, 1:]
        iflat = input_.contiguous().view(N, C, -1)
        tflat = target.contiguous().view(N, C, -1)
        weight = 1 / (tflat.sum(dim=2)**2 + self.smooth)
        intersection = weight * (iflat * tflat).sum(dim=2)
        union = iflat.sum(dim=2) + tflat.sum(dim=2)
        dice = (2. * weight * intersection + self.smooth) / (weight * union + self.smooth)
        if self.reduce == 'mean':
            loss = 1 - dice.mean()
        elif self.reduce == 'sum':
            loss = N - dice.sum()

        return loss



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
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