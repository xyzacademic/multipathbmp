import numpy as np
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Combine data')

parser.add_argument('--net', default='ours', type=str, help='which network')
args = parser.parse_args()

# df = pd.read_csv('statistic/mask_stats.csv')
# patient = df['Name'].values.tolist()

save_path = '%s/second_phase_data' % args.net
os.chdir(save_path)
data_list = [0, 1, 2, 3, 4]
view_list = ['XY_12', 'XY_3', 'XY_123', 'ZX_12', 'ZX_3', 'ZX_123', 'ZY_12', 'ZY_3', 'ZY_123']

data = [[np.load('%d_%s.npy' % (i, j))[:, np.newaxis, np.newaxis, :, :, :, :] for j in view_list] for i in data_list]

for i in range(len(data)):
    data[i] = np.concatenate(data[i], axis=2)

channels = 3
data = np.concatenate(data, axis=1)
data = data.reshape((-1, 9, channels, 192, 224, 192))
data = data.reshape((data.shape[0], channels * 9, 192, 224, 192))
print(data.shape)
np.save('train_data.npy', data)
del data

# data = [np.load('5_%s.npy' % j)[:, np.newaxis, :, :, :, :] for j in view_list]
# data = np.concatenate(data, axis=1)
# data = data.reshape((data.shape[0], channels * 9, 192, 224, 192))
# np.save('test_data.npy', data)
# print(data.shape)
# del data
# os.chdir(os.pardir)
# os.chdir(os.pardir)








