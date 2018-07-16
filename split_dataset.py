import random
import shutil
from os.path import join
train_ratio = 0.9

def read_txt(file_list):
    data_list = []
    with open(file_list) as f:
        for line in f:
            fn = line.split()
            data_list.append(fn[0])
    return data_list

def make_list(MEL_DIR):
    data_list = read_txt(join(MEL_DIR, 'train.txt'))
    shutil.copy2(join(MEL_DIR, 'train.txt'), join(MEL_DIR, 'train_all.txt'))

    num_train_size = int(len(data_list) * train_ratio)
    data_list.sort()
    random.Random(4).shuffle(data_list)

    train_list = data_list[:num_train_size]
    valid_list = data_list[num_train_size:]
    train_list.sort()
    valid_list.sort()
    with open(join(MEL_DIR, 'train.txt'), 'w') as f:
        for item in train_list:
            f.write("%s\n" % (item))
    with open(join(MEL_DIR, 'valid.txt'), 'w') as f:
        for item in valid_list:
            f.write("%s\n" % (item))

    return train_list, valid_list

if __name__ == '__main__':
    DATA_ROOT = '../data'

    TARGET = ['Happy', 'Sad', 'Angry']

    for target in TARGET :
        MEL_DIR = join(DATA_ROOT, 'dtw_{}'.format(target))
        make_list(MEL_DIR)