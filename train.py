"""Trainining script for seq2seq emotional conversion model.

usage: train.py [options] <target>

options:
    --data_root=<dir>            Directory contains preprocessed features [default: ../data].
    --checkpoint_dir=<dir>       Directory where to save model checkpoints.
    --checkpoint_path=<path>     Restore model from checkpoint path if given.
    --log_event_path=<path>      Log event path.
    --train_dir=<path>           Directory contains preprocessed dtw features.
    --train-seq2seq-only         Train only seq2seq model.
    --train-postnet-only         Train only postnet model.
    -h, --help                   Show this help message and exit
"""
import os
import numpy as np
from Trainer import Trainer
from torch.utils.data import DataLoader
from os.path import join
from datetime import datetime
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from torch.utils.data.sampler import Sampler
from model import EVCModel, NEU2EMO, MEL2LIN
from docopt import docopt
import torch
from hparams import hparams
from tensorboardX import SummaryWriter
import random
import torch.backends.cudnn as cudnn

use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False

global_step = 0
global_epoch = 1

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

class TLinearSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, data_type):
        super(TLinearSpecDataSource, self).__init__(data_root, 0, data_type)

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
    def __init__(self, S_Mel, T_Mel, T_Linear):
        self.S_Mel = S_Mel
        self.T_Mel = T_Mel
        self.T_Linear = T_Linear

    def __getitem__(self, idx):
        return self.S_Mel[idx], self.T_Mel[idx], self.T_Linear[idx]

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
    S_Mbatch = torch.FloatTensor(S_Mel)

    T_Mel = np.array([_pad_2d(x[1], max_length) for x in batch], dtype=np.float32)
    T_Mbatch = torch.FloatTensor(T_Mel)

    T_Linear = np.array([_pad_2d(x[2], max_length) for x in batch], dtype=np.float32)
    T_Lbatch = torch.FloatTensor(T_Linear)

    lengths = torch.LongTensor(lengths)
    return S_Mbatch, T_Mbatch, T_Lbatch, lengths


def load_checkpoint(path, model, optimizer):
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])

    #optimizer_state = checkpoint["optimizer"]
    print("Load optimizer state from {}".format(path))
    optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model, optimizer, global_step, global_epoch

def build_model(train_seq2seq, train_postnet, device, hparams):
    h_c = hparams.converter_channels
    h_p = hparams.postnet_channels
    k = hparams.kernel_size
    out_dim = int(hparams.fft_size / 2 + 1)
    if train_seq2seq:
        seq2seq = NEU2EMO(in_dim=hparams.num_mels, dropout=hparams.dropout, preattention=[(h_c, k, 1), (h_c, k, 3)],
            convolutions=[(h_c, k, 1), (h_c, k, 3), (h_c, k, 9), (h_c, k, 27), (h_c, k, 1)]).to(device)
    if train_postnet:
        postnet = MEL2LIN(in_dim=hparams.num_mels, out_dim=out_dim, dropout=hparams.dropout,
            convolutions=[(h_p, k, 1), (h_p, k, 3), (2 * h_p, k, 1), (2 * h_p, k, 3)]).to(device)
    model = EVCModel(seq2seq, postnet, mel_dim=hparams.num_mels, linear_dim=out_dim)
    return model

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)

    target = args["<target>"]
    data_root = args["--data_root"]
    checkpoint_dir = args["--checkpoint_dir"]
    checkpoint_path = args["--checkpoint_path"]
    log_event_path = args["--log_event_path"]
    train_dir = args["--train_dir"]
    # Which model to be trained
    train_seq2seq = args["--train-seq2seq-only"]
    train_postnet = args["--train-postnet-only"]
    # train both if not specified
    if not train_seq2seq and not train_postnet:
        print("Training whole model")
        train_seq2seq, train_postnet = True, True
    if train_seq2seq:
        print("Training seq2seq model")
    elif train_postnet:
        print("Training postnet model")
    else:
        assert False, "must be specified wrong args"

    if log_event_path is None:
        log_event_path = join('./log', target, str(datetime.now()).replace(" ", "_"))
        os.makedirs(log_event_path, exist_ok=True)
    if checkpoint_dir is None:
        checkpoint_dir = join('./checkpoint', target, str(datetime.now()).replace(" ", "_"))
        os.makedirs(checkpoint_dir, exist_ok=True)
    if train_dir is None:
        train_dir = join(data_root, 'dtw_{}'.format(target))  # ./data/dtw_Sad

    trn_S_Mel = FileSourceDataset(SMelSpecDataSource(train_dir, 'train'))
    trn_T_Mel = FileSourceDataset(TMelSpecDataSource(train_dir, 'train'))
    trn_T_Linear = FileSourceDataset(TLinearSpecDataSource(train_dir, 'train'))
    val_S_Mel = FileSourceDataset(SMelSpecDataSource(train_dir, 'valid'))
    val_T_Mel = FileSourceDataset(TMelSpecDataSource(train_dir, 'valid'))
    val_T_Linear = FileSourceDataset(TLinearSpecDataSource(train_dir, 'valid'))

    frame_lengths = trn_S_Mel.file_data_source.frame_lengths

    sampler = PartialyRandomizedSimilarTimeLengthSampler(frame_lengths, batch_size=hparams.batch_size)

    train_dataset = PyTorchDataset(trn_S_Mel, trn_T_Mel, trn_T_Linear)
    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, num_workers=hparams.num_workers, sampler=sampler, collate_fn=collate_fn, pin_memory=True)
    valid_dataset = PyTorchDataset(val_S_Mel, val_T_Mel, val_T_Linear)
    valid_loader = DataLoader(valid_dataset, batch_size=10, num_workers=hparams.num_workers, collate_fn=collate_fn)


    device = torch.device("cuda" if use_cuda else "cpu")
    model = build_model(train_seq2seq, train_postnet, device, hparams)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.initial_learning_rate, betas=(hparams.adam_beta1, hparams.adam_beta2),
                                 eps=hparams.adam_eps, weight_decay=hparams.weight_decay, amsgrad=False)
    if checkpoint_path is not None:
        model, optimizer, global_step, global_epoch = load_checkpoint(checkpoint_path, model, optimizer)

    writer = SummaryWriter(log_dir=log_event_path)
    trainer = Trainer(model, train_loader, valid_loader=valid_loader, optimizer=optimizer, writer=writer, checkpoint_dir=checkpoint_dir, device=device, hparams=hparams)
    try:
        trainer.train(train_seq2seq=train_seq2seq, train_postnet=train_postnet, global_epoch=global_epoch, global_step=global_step)
    except:
        print()

