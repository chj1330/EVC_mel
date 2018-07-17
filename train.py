import os
import numpy as np
from Trainer import Trainer
from torch.utils.data import DataLoader
from os.path import join
from datetime import datetime
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from torch.utils.data.sampler import Sampler
from model import ConvBlock
import time
import torch
import argparse
from model import *
import random
import torch.backends.cudnn as cudnn
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False

class _NPYDataSource(FileDataSource):
    def __init__(self, train_dir, col, data_type):
        self.train_dir = train_dir
        self.col = col
        self.frame_lengths = []
        self.data_type = data_type

    def collect_files(self):
        meta = join(self.train_dir, "{}.txt".format(self.data_type))
        with open(meta, "rb") as f:
            lines = f.readlines()
        self.frame_lengths = list(map(lambda l: int(l.decode("utf-8").split("|")[3]), lines))

        paths = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(self.train_dir, f), paths))

        return paths

    def collect_features(self, path):
        return np.load(path)

class SMelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, data_type):
        super(SMelSpecDataSource, self).__init__(data_root, 1, data_type)

class TMelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, data_type):
        super(TMelSpecDataSource, self).__init__(data_root, 2, data_type)


class PartialyRandomizedSimilarTimeLengthSampler(Sampler):
    """Partially randmoized sampler

    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batchs
    """

    def __init__(self, lengths, batch_size=16, batch_group_size=None,
                 permutate=True):
        self.lengths, self.sorted_indices = torch.sort(torch.LongTensor(lengths), descending=True)
        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 32, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0
        self.permutate = permutate

    def __iter__(self):
        indices = self.sorted_indices.clone()
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            random.shuffle(indices[s:e])

        # Permutate batches
        if self.permutate:
            perm = np.arange(len(indices[:e]) // self.batch_size)
            random.shuffle(perm)
            indices[:e] = indices[:e].view(-1, self.batch_size)[perm, :].view(-1)

        # Handle last elements
        s += batch_group_size
        if s < len(indices):
            random.shuffle(indices[s:])

        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)


class PyTorchDataset(object):
    def __init__(self, S_Mel, T_Mel):
        self.S_Mel = S_Mel
        self.T_Mel = T_Mel

    def __getitem__(self, idx):
        return self.S_Mel[idx], self.T_Mel[idx]

    def __len__(self):
        return len(self.T_Mel)

def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


def collate_fn(batch):
    lengths = [len(x[0]) for x in batch]
    max_length = max(lengths)

    S_Mel = np.array([_pad_2d(x[0], max_length) for x in batch], dtype=np.float32)
    S_batch = torch.FloatTensor(S_Mel)

    T_Mel = np.array([_pad_2d(x[1], max_length) for x in batch], dtype=np.float32)
    T_batch = torch.FloatTensor(T_Mel)

    lengths = torch.LongTensor(lengths)
    return S_batch, T_batch, lengths

if __name__ == "__main__":
    GPU_USE = 1
    DEVICE = 'cuda'  # 0 : gpu0, 1 : gpu1, ...
    EPOCH = 20000
    BATCH_SIZE = 16
    LEARN_RATE = 0.1
    DATA_ROOT = '../data'
    TARGET = 'Sad'
    #LOG_DIR = join('./log_convAE', TARGET, '{}_lf0'.format(FEAT_TYPE), str(datetime.now()).replace(" ", "_"))
    LOG_DIR = join('./log', TARGET, str(datetime.now()).replace(" ", "_"))
    TRN_DIR = join(DATA_ROOT, 'dtw_{}'.format(TARGET))  # ./data/dtw_Sad
    CHECKPOINT_DIR = join('./checkpoint', TARGET, str(datetime.now()).replace(" ", "_"))
    NUM_WORKERS = 8
    BINARY_DIVERGENCE_WEIGHT = 0.1
    MASKED_LOSS_WEIGHT = 0.5
    CHECKPOINT_INTERVAL = 100 # Save checkpoint
    EVAL_INTERVAL = 100 # Plot melspectrogram

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='log directory')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR, help='checkpoint directory')
    parser.add_argument('--device', type=str, default=DEVICE, help='which device?')
    parser.add_argument('--gpu_use', type=int, default=GPU_USE, help='GPU enable? 0 : cpu, 1 : gpu')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='how many workers?')
    parser.add_argument('--epoch', type=int, default=EPOCH, help='how many epoch?')
    parser.add_argument('--checkpoint_inverval', type=int, default=CHECKPOINT_INTERVAL, help='checkpoint interval')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='how many batch?')
    parser.add_argument('--learn_rate', type=float, default=LEARN_RATE, help='learning rate')
    parser.add_argument('--binary_divergence_weight', type=float, default=BINARY_DIVERGENCE_WEIGHT, help='binary divergence weight')
    parser.add_argument('--masked_loss_weight', type=float, default=MASKED_LOSS_WEIGHT, help='masked loss weight')
    parser.add_argument('--eval_interval', type=float, default=EVAL_INTERVAL, help='eval interval')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT, help='data directory')
    parser.add_argument('--target', type=str, default=TARGET, help='target emotion')
    parser.add_argument('--train_dir', type=str, default=TRN_DIR, help='training data directory')

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    trn_S_Mel = FileSourceDataset(SMelSpecDataSource(TRN_DIR, 'train'))
    trn_T_Mel = FileSourceDataset(TMelSpecDataSource(TRN_DIR, 'train'))
    val_S_Mel = FileSourceDataset(SMelSpecDataSource(TRN_DIR, 'valid'))
    val_T_Mel = FileSourceDataset(TMelSpecDataSource(TRN_DIR, 'valid'))

    frame_lengths = trn_S_Mel.file_data_source.frame_lengths

    sampler = PartialyRandomizedSimilarTimeLengthSampler(frame_lengths, batch_size=args.batch_size)

    train_dataset = PyTorchDataset(trn_S_Mel, trn_T_Mel)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler, collate_fn=collate_fn, pin_memory=True)
    valid_dataset = PyTorchDataset(val_S_Mel, val_T_Mel)
    valid_loader = DataLoader(valid_dataset, batch_size=10, num_workers=args.num_workers, collate_fn=collate_fn)


    device = torch.device("cuda" if args.gpu_use else "cpu")
    h = 256  # hidden dim (channels)
    k = 3  # kernel size
    # Initialize
    model = ConvBlock(in_dim=80, dropout=0.05, preattention=[(h, k, 1), (h, k, 3)],
        convolutions=[(h, k, 1), (h, k, 3), (h, k, 9), (h, k, 27), (h, k, 1)]).to(device)

    trainer = Trainer(model=model, train_loader=train_loader, valid_loader=valid_loader, device=device, args=args)
    try:
        trainer.train()
    except:
        print()

