import numpy as np
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='Combine data')

parser.add_argument('--net', default='ours', type=str, help='which network')
args = parser.parse_args()

# label = np.concatenate([np.load('data/KES_label.npy'), np.load('data/MCW_label.npy')], axis=0)
label = np.load('data5/train_label.npy')
experiment_path = args.net

data = np.load('%s/second_phase_data/train_data.npy' % experiment_path)

for k in range(5):
    train_index = []
    test_index = []
    for i in range(label.shape[0]):
        if i % 5 == k:
            test_index.append(i)
        else:
            train_index.append(i)

    if not os.path.exists('%s/second_data%d' % (experiment_path, k)):
        os.makedirs('%s/second_data%d' % (experiment_path, k))
    np.save('%s/second_data%d/train_data.npy' % (experiment_path, k), data[train_index])
    np.save('%s/second_data%d/train_label.npy' % (experiment_path, k), label[train_index])
    np.save('%s/second_data%d/test_data.npy' % (experiment_path, k), data[test_index])
    np.save('%s/second_data%d/test_label.npy' % (experiment_path, k), label[test_index])


# k = 5
# if not os.path.exists('%s/second_data%d' % (experiment_path, k)):
#     os.makedirs('%s/second_data%d' % (experiment_path, k))
#
# shutil.copy('%s/second_phase_data/train_data.npy' % experiment_path, '%s/second_data%d/train_data.npy' % (experiment_path, k))
# shutil.copy('%s/second_phase_data/test_data.npy' % experiment_path, '%s/second_data%d/test_data.npy' % (experiment_path, k))
# shutil.copy('data5/train_label.npy', '%s/second_data%d/train_label.npy' % (experiment_path, k))
# shutil.copy('data5/test_label.npy', '%s/second_data%d/test_label.npy' % (experiment_path, k))
