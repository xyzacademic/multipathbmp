import numpy as np





def dice_coef(pred, target, dims=(1, 2, 3)):
    # assert pred.shape == target.shape
    a = pred + target
    overlap = (pred * target).sum(axis=dims) * 2
    union = a.sum(axis=dims)
    epsilon = 0.0001
    dice = overlap / (union + epsilon)

    return dice

def IOU(pred, target, dims=(1, 2, 3)):
    # assert pred.shape == target.shape
    a = pred + target
    overlap = (pred * target).sum(axis=dims)
    union = (a > 0).sum(axis=dims)
    epsilon = 0.0001
    iou = overlap / (union + epsilon)

    return iou

def Recall(pred, target, dims=(1, 2, 3)):
    a = pred * target
    epsilon = 0.0001
    return a.sum(axis=dims) / (target.sum(axis=dims) + epsilon)

def Precision(pred, target, dims=(1, 2, 3)):
    a = pred * target
    epsilon = 0.0001
    return a.sum(axis=dims) / (pred.sum(axis=dims) + epsilon)

def get_dict():
    em = {}

    em['dice'] = dice_coef
    em['iou'] = IOU
    em['recall'] = Recall
    em['precision'] = Precision


    return em