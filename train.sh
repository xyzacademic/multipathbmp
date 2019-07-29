#!/bin/sh


cd ..
#
# 2d slices training
#data 0
python train_2d.py --gpu -1 --view XY --norm-axis 12 --net ours --epoch 50 --data 0 --lr 0.01
python train_2d.py --gpu -1 --view XY --norm-axis 3 --net ours --epoch 50 --data 0 --lr 0.01
python train_2d.py --gpu -1 --view ZY --norm-axis 12 --net ours --epoch 50 --data 0 --lr 0.01
python train_2d.py --gpu -1 --view ZY --norm-axis 3 --net ours --epoch 50 --data 0 --lr 0.01
python train_2d.py --gpu -1 --view ZX --norm-axis 12 --net ours --epoch 50 --data 0 --lr 0.01
python train_2d.py --gpu -1 --view ZX --norm-axis 3 --net ours --epoch 50 --data 0 --lr 0.01
python train_2d.py --gpu -1 --view XY --norm-axis 123 --net ours --epoch 50 --data 0 --lr 0.01
python train_2d.py --gpu -1 --view ZY --norm-axis 123 --net ours --epoch 50 --data 0 --lr 0.01
python train_2d.py --gpu -1 --view ZX --norm-axis 123 --net ours --epoch 50 --data 0 --lr 0.01



# 2d slices prediction
##data 0
python predict_2d.py --gpu -1 --view XY --norm-axis 12 --net ours --epoch 50 --data 0
python predict_2d.py --gpu -1 --view XY --norm-axis 3 --net ours --epoch 50 --data 0
python predict_2d.py --gpu -1 --view ZY --norm-axis 12 --net ours --epoch 50 --data 0
python predict_2d.py --gpu -1 --view ZY --norm-axis 3 --net ours --epoch 50 --data 0
python predict_2d.py --gpu -1 --view ZX --norm-axis 12 --net ours --epoch 50 --data 0
python predict_2d.py --gpu -1 --view ZX --norm-axis 3 --net ours --epoch 50 --data 0
python predict_2d.py --gpu -1 --view XY --norm-axis 123 --net ours --epoch 50 --data 0
python predict_2d.py --gpu -1 --view ZY --norm-axis 123 --net ours --epoch 50 --data 0
python predict_2d.py --gpu -1 --view ZX --norm-axis 123 --net ours --epoch 50 --data 0

## 3d data prepare
##data 0
python prepare_secondphase_2d.py --gpu -1 --view XY --norm-axis 12 --net ours --epoch 50 --data 0
python prepare_secondphase_2d.py --gpu -1 --view XY --norm-axis 3 --net ours --epoch 50 --data 0
python prepare_secondphase_2d.py --gpu -1 --view ZY --norm-axis 12 --net ours --epoch 50 --data 0
python prepare_secondphase_2d.py --gpu -1 --view ZY --norm-axis 3 --net ours --epoch 50 --data 0
python prepare_secondphase_2d.py --gpu -1 --view ZX --norm-axis 12 --net ours --epoch 50 --data 0
python prepare_secondphase_2d.py --gpu -1 --view ZX --norm-axis 3 --net ours --epoch 50 --data 0
python prepare_secondphase_2d.py --gpu -1 --view XY --norm-axis 123 --net ours --epoch 50 --data 0
python prepare_secondphase_2d.py --gpu -1 --view ZY --norm-axis 123 --net ours --epoch 50 --data 0
python prepare_secondphase_2d.py --gpu -1 --view ZX --norm-axis 123 --net ours --epoch 50 --data 0


#combine data
python combine_data.py --net ours

#split second phase dataset
python split_second_phase_data.py --net ours


#second phase train
python second_phase_train.py --gpu -1 --view XY --norm-axis 12 --net ours --epoch 50 --data 0 --lr 0.01

#second phase predict
python second_phase_predict.py --gpu -1 --view XY --norm-axis 12 --net ours --epoch 50 --data 0 --lr 0.01

#combine prediction
python combine_prediction.py --net ours

