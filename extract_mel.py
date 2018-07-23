import os
import numpy as np
from os.path import join
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from model import EVCModel, NEU2EMO, MEL2LIN
import torch
import audio
import argparse
import torch.backends.cudnn as cudnn
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False


if __name__ == "__main__":
    GPU_USE = 0
    DATA_ROOT = '../data'
    TARGET = 'Happy'
    SOURCE = 'Neutral'
    RESULT_DIR = join('./result', TARGET)
    TRN_DIR = join(DATA_ROOT, 'dtw_{}'.format(TARGET))  # ./data/dtw_Sad
    ORIGIN_DIR = join(DATA_ROOT, 'emotion')  # ./data/emotion
    CHECKPOINT = "./checkpoint/Happy/2018-07-17_20:50:20.548680/checkpoint_epoch000010000.pth"

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR, help='result directory')
    parser.add_argument('--gpu_use', type=int, default=GPU_USE, help='GPU enable? 0 : cpu, 1 : gpu')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT, help='data directory')
    parser.add_argument('--target', type=str, default=TARGET, help='target emotion')
    parser.add_argument('--source', type=str, default=SOURCE, help='source emotion')
    parser.add_argument('--train_dir', type=str, default=TRN_DIR, help='training data directory')
    parser.add_argument('--origin_dir', type=str, default=ORIGIN_DIR, help='training data directory')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT, help='checkpoint path')


    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    device = torch.device("cuda" if args.gpu_use else "cpu")

    h = 256  # hidden dim (channels)
    k = 3  # kernel size
    # Initialize
    train_seq2seq = True
    train_postnet = True

    if train_seq2seq:
        NEU2EMO = NEU2EMO(in_dim=80, dropout=0.05, preattention=[(h, k, 1), (h, k, 3)], convolutions=[(h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27), (h, k, 1)]).to(device)
    if train_postnet:
        MEL2LIN = MEL2LIN(in_dim=80, out_dim=513, dropout=0.05, convolutions=[(h, k, 1), (h, k, 3), (2 * h, k, 1), (2 * h, k, 3)]).to(device)
    model = EVCModel(NEU2EMO, MEL2LIN, mel_dim=80, linear_dim=513)

    checkpoint = torch.load(args.checkpoint)

    model.load_state_dict(checkpoint['state_dict'])

    model.eval()


    with open(join(args.train_dir, 'test.txt'), encoding='utf-8') as f:
        for line in f:
            fn = line.split()
            l = fn[0].split("|")
            melX = join(args.origin_dir, l[1])
            melX = np.load(melX)
            melX = torch.from_numpy(melX).unsqueeze(0).to(device)
            mel_output, _ = model(melX)

            mel_output = mel_output[0].data.cpu().numpy()
            path = join(args.result_dir, l[1])
            np.save(path, mel_output)

            print('%s' % l[1])


