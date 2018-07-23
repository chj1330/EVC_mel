import random
import shutil
from os.path import join
train_ratio = 0.85
valid_ratio = 0.1
test_ratio = 0.05
def read_txt(file_list):
    ema_list = []
    emb_list = []
    emc_list = []
    emd_list = []
    eme_list = []
    idx = 0
    with open(file_list) as f:
        for line in f:
            fn = line.split()
            if idx < 100 :
                ema_list.append(fn[0])
            elif idx < 200 :
                emb_list.append(fn[0])
            elif idx < 300 :
                emc_list.append(fn[0])
            elif idx < 400 :
                emd_list.append(fn[0])
            elif idx < 500:
                eme_list.append(fn[0])
            idx += 1
    return ema_list, emb_list, emc_list, emd_list, eme_list

def make_list(MEL_DIR):
    a, b, c, d, e = read_txt(join(MEL_DIR, 'train.txt'))
    shutil.copy2(join(MEL_DIR, 'train.txt'), join(MEL_DIR, 'all.txt'))

    train_size = int(len(a) * train_ratio)
    valid_size = int(len(a) * valid_ratio)
    a.sort()
    b.sort()
    c.sort()
    d.sort()
    e.sort()
    random.Random(4).shuffle(a)
    random.Random(4).shuffle(b)
    random.Random(4).shuffle(c)
    random.Random(4).shuffle(d)
    random.Random(4).shuffle(e)

    train_list = a[:train_size] + a[:train_size] + c[:train_size] + d[:train_size] + e[:train_size]
    valid_list = a[train_size:train_size+valid_size] + b[train_size:train_size+valid_size] + c[train_size:train_size+valid_size] + d[train_size:train_size+valid_size] + e[train_size:train_size+valid_size]
    test_list = a[train_size+valid_size:] + b[train_size+valid_size:] + c[train_size+valid_size:] + d[train_size+valid_size:] + e[train_size+valid_size:]
    train_list.sort()
    valid_list.sort()
    test_list.sort()
    with open(join(MEL_DIR, 'train.txt'), 'w') as f:
        for item in train_list:
            f.write("%s\n" % (item))
    with open(join(MEL_DIR, 'valid.txt'), 'w') as f:
        for item in valid_list:
            f.write("%s\n" % (item))
    with open(join(MEL_DIR, 'test.txt'), 'w') as f:
        for item in test_list:
            f.write("%s\n" % (item))



if __name__ == '__main__':
    DATA_ROOT = '../data'

    TARGET = ['Happy']
    #TARGET = ['Sad']
    for target in TARGET :
        MEL_DIR = join(DATA_ROOT, 'dtw_{}'.format(target))
        make_list(MEL_DIR)